#include <malloc.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include <H5PLextern.h>
#include <H5Zpublic.h>
#include <hdf5.h>

#include "span.hpp"

#include "charls/charls.h"
#include "threadpool.h"
ThreadPool* filter_pool = nullptr;

#include <future>

using std::vector;

namespace {

// Temporary unofficial filter ID
const H5Z_filter_t H5Z_FILTER_JPEGLS = 32012;

std::tuple<int, size_t, int>
getParams(const size_t cd_nelmts, const unsigned int cd_values[]) {
    if (cd_nelmts <= 3 || cd_values[0] == 0) {
        return {-1, 0, 0};
    }

    int length = cd_values[0];
    size_t nblocks = cd_values[1];
    int typesize = cd_values[2];

    return {length, nblocks, typesize};
}

}  // namespace

size_t
codec_filter(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[], size_t nbytes,
             size_t* buf_size, void** buf) {
    const auto [length, nblocks, typesize] = getParams(cd_nelmts, cd_values);

    if (length == -1) {
        std::cerr << "Error: Incorrect number of filter parameters specified. Aborting.\n";
        return -1;
    }

    const size_t subchunks = std::min(size_t(24), nblocks);
    const size_t lblocks = nblocks / subchunks;
    const size_t header_size = 4 * subchunks;
    const size_t remainder = nblocks - lblocks * subchunks;

    if (flags & H5Z_FLAG_REVERSE) {
        char err_msg[256];

        /* Input */
        auto in_buf = static_cast<unsigned char*>(realloc(*buf, nblocks * length * typesize * 2));
        *buf = in_buf;

        std::vector<uint32_t> block_size(subchunks);
        // Extract header
        std::copy_n(reinterpret_cast<uint32_t*>(in_buf), block_size.size(), block_size.begin());

        std::vector<uint32_t> offset(subchunks);
        offset[0] = 0;
        std::partial_sum(block_size.begin(), block_size.end() - 1, offset.begin() + 1);

        const tcb::span<unsigned char> in_buf_span{in_buf, offset.back() + block_size.back()};

        vector<vector<unsigned char>> tbuf(subchunks);

// Make a copy of the compressed buffer. Required because we
// now realloc in_buf.
#pragma omp parallel for schedule(guided)
        for (size_t block = 0; block < subchunks; block++) {
            const auto in = in_buf_span.subspan(offset[block], block_size[block]);
            auto& out = tbuf[block];

            out.insert(out.begin(), in.begin(), in.end());
        }

        vector<std::future<void>> futures;
        for (size_t block = 0; block < subchunks; block++) {
            futures.emplace_back(filter_pool->enqueue([&, block] {
                size_t own_blocks = (block < remainder ? 1 : 0) + lblocks;
                CharlsApiResultType ret = JpegLsDecode(
                    in_buf + typesize * length *
                                 ((block < remainder) ? block * (lblocks + 1)
                                                      : (remainder * (lblocks + 1) +
                                                         (block - remainder) * lblocks)),
                    typesize * length * own_blocks, tbuf[block].data(), block_size[block], nullptr,
                    err_msg);
                if (ret != CharlsApiResultType::OK) {
                    fprintf(stderr, "JPEG-LS error %d: %s\n", ret, err_msg);
                }
            }));
        }
        for (auto& future : futures) {
            future.wait();
        }

        *buf_size = nblocks * length * typesize;

        return *buf_size;

    } else {
        /* Output */

        auto in_buf = reinterpret_cast<unsigned char*>(*buf);

        std::vector<uint32_t> block_size(subchunks);
        std::vector<std::vector<unsigned char>> local_out(subchunks);

#pragma omp parallel for schedule(guided)
        for (size_t block = 0; block < subchunks; block++) {
            const size_t own_blocks = (block < remainder ? 1 : 0) + lblocks;
            auto& local_buf = local_out[block];

            const auto reserved_size = own_blocks * length * typesize;
            local_buf.resize(reserved_size + 8192);

            auto params = [&]() -> const JlsParameters {
                auto params = JlsParameters();
                params.width = length;
                params.height = own_blocks;
                params.bitsPerSample = typesize * 8;
                params.components = 1;
                return params;
            }();

            size_t csize;
            char err_msg[256];
            const CharlsApiResultType ret = JpegLsEncode(
                local_buf.data(), local_buf.size(), &csize,
                in_buf + typesize * length *
                             ((block < remainder)
                                  ? block * (lblocks + 1)
                                  : (remainder * (lblocks + 1) + (block - remainder) * lblocks)),
                reserved_size, &params, err_msg);
            if (ret != CharlsApiResultType::OK) {
                std::cerr << "JPEG-LS error: " << err_msg << '\n';
            }
            local_buf.resize(csize);
        }

        const auto compressed_size =
            std::accumulate(local_out.begin(), local_out.end(), header_size,
                            [](const auto& a, const auto& b) -> size_t { return a + b.size(); });

        if (compressed_size > nbytes) {
            in_buf = reinterpret_cast<unsigned char*>(realloc(*buf, compressed_size));
            *buf = in_buf;
        }

        auto header = reinterpret_cast<uint32_t*>(in_buf);
#pragma omp parallel for schedule(guided)
        for (size_t block = 0; block < subchunks; block++) {
            const auto offset = std::accumulate(
                local_out.begin(), local_out.begin() + block, header_size,
                [](const auto& a, const auto& b) -> size_t { return a + b.size(); });

            const auto& local_buf = local_out[block];

            // Write header
            header[block] = local_buf.size();

            // Write payload
            std::copy(local_buf.begin(), local_buf.end(), in_buf + offset);
        }

        *buf_size = compressed_size;

        return compressed_size;
    }
}

herr_t h5jpegls_set_local(hid_t dcpl, hid_t type, hid_t) {  // NOLINT
    const auto [r, flags,
                values] = [&]() -> std::tuple<herr_t, unsigned int, std::vector<unsigned int>> {
        unsigned int flags;
        std::vector<unsigned int> values(8);
        size_t nelements = values.size();

        const auto r = H5Pget_filter_by_id(dcpl, H5Z_FILTER_JPEGLS, &flags, &nelements,
                                           values.data(), 0, nullptr, nullptr);

        if (r < 0) {
            return {r, 0, {}};
        }

        values.resize(nelements);
        return {r, flags, values};
    }();

    if (r < 0) {
        return -1;
    }

    hsize_t chunkdims[32];
    const int ndims = H5Pget_chunk(dcpl, 32, chunkdims);
    if (ndims < 0) {
        return -1;
    }

    const bool byte_mode = values.size() > 0 && values[0] != 0;

    constexpr unsigned int minus_one = -1;

    auto cb_values = [&]() -> const std::array<unsigned int, 4> {
        unsigned int length = chunkdims[ndims - 1];
        unsigned int nblocks = (ndims == 1) ? 1 : chunkdims[ndims - 2];

        unsigned int typesize = H5Tget_size(type);
        if (typesize == 0) {
            return {minus_one, 0, 0};
        }

        H5T_class_t classt = H5Tget_class(type);
        if (classt == H5T_ARRAY) {
            hid_t super_type = H5Tget_super(type);
            typesize = H5Tget_size(super_type);
            H5Tclose(super_type);
        }

        if (byte_mode) {
            typesize = 1;
            length *= typesize;
        }

        return {length, nblocks, typesize};
    }();

    if (cb_values[0] == minus_one) {
        return -1;
    }

    // nelements = 3; // TODO: update if we accept #subchunks
    {
        const auto r =
            H5Pmodify_filter(dcpl, H5Z_FILTER_JPEGLS, flags, cb_values.size(), cb_values.data());

        if (r < 0) {
            return -1;
        }
    }

    return 1;
}

const H5Z_class2_t H5Z_JPEGLS[1] = {{
    H5Z_CLASS_T_VERS,                                      /* H5Z_class_t version */
    H5Z_FILTER_JPEGLS,                                     /* Filter id number */
    1,                                                     /* encoder_present flag (set to true) */
    1,                                                     /* decoder_present flag (set to true) */
    "HDF5 JPEG-LS filter v0.2",                            /* Filter name for debugging */
    nullptr,                                               /* The "can apply" callback     */
    static_cast<H5Z_set_local_func_t>(h5jpegls_set_local), /* The "set local" callback */
    static_cast<H5Z_func_t>(codec_filter),                 /* The actual filter function */
}};

H5PL_type_t H5PLget_plugin_type() {  // NOLINT
    return H5PL_TYPE_FILTER;
}
const void* H5PLget_plugin_info() {  // NOLINT
    return H5Z_JPEGLS;
}

__attribute__((constructor)) void
init_threadpool() {
    int threads = 0;
    char* envvar = getenv("HDF5_FILTER_THREADS");
    if (envvar != nullptr) {
        threads = atoi(envvar);
    }
    if (threads <= 0) {
        threads = std::min(std::thread::hardware_concurrency(), 8u);
    }
    filter_pool = new ThreadPool(threads);
}

__attribute__((destructor)) void
destroy_threadpool() {
    delete filter_pool;
}
