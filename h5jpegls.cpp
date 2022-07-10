#include <malloc.h>
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

#include "jpegls-filter.h"

#include "charls/charls.h"
#include "threadpool.h"
ThreadPool* filter_pool = nullptr;

#include <future>

using std::vector;

#define VISIBLE __attribute__ ((visibility ("default")))

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

VISIBLE
size_t
codec_filter(unsigned int flags, size_t cd_nelmts, const unsigned int cd_values[], size_t nbytes,
             size_t* buf_size, void** buf) {
    const auto [length, nblocks, typesize] = getParams(cd_nelmts, cd_values);

    if (length == -1) {
        std::cerr << "Error: Incorrect number of filter parameters specified. Aborting.\n";
        return -1;
    }

    const jpegls::subchunk_config_t config(length, nblocks, typesize);

    if (flags & H5Z_FLAG_REVERSE) {
        const size_t subchunks = config.subchunks;
        const size_t lblocks = config.lblocks;
        const size_t header_size = config.header_size;
        const size_t remainder = config.remainder;

        char err_msg[256];

        filter_pool->lock_buffers();
        /* Input */
        auto in_buf = static_cast<unsigned char*>(realloc(*buf, nblocks * length * typesize * 2));
        *buf = in_buf;

        uint32_t block_size[subchunks];
        uint32_t offset[subchunks];
        // Extract header
        memcpy(block_size, in_buf, subchunks * sizeof(uint32_t));

        offset[0] = 0;
        uint32_t coffset = 0;
        for (size_t block = 1; block < subchunks; block++) {
            coffset += block_size[block - 1];
            offset[block] = coffset;
        }

        unsigned char* tbuf[subchunks];
        vector<std::future<void>> futures;
        // Make a copy of the compressed buffer. Required because we
        // now realloc in_buf.
        for (size_t block = 0; block < subchunks; block++) {
            futures.emplace_back(filter_pool->enqueue([&, block] {
                tbuf[block] =
                    filter_pool->get_global_buffer(block, length * nblocks * typesize + 512);
                memcpy(tbuf[block], in_buf + header_size + offset[block], block_size[block]);
            }));
        }
        // must wait for copies to complete, otherwise having
        // threads > subchunks could lead to a decompressor overwriting in_buf
        for (auto& future : futures) {
            future.wait();
        }

        for (size_t block = 0; block < subchunks; block++) {
            futures.emplace_back(filter_pool->enqueue([&, block] {
                size_t own_blocks = (block < remainder ? 1 : 0) + lblocks;
                CharlsApiResultType ret = JpegLsDecode(
                    in_buf + typesize * length *
                                 ((block < remainder) ? block * (lblocks + 1)
                                                      : (remainder * (lblocks + 1) +
                                                         (block - remainder) * lblocks)),
                    typesize * length * own_blocks, tbuf[block], block_size[block], nullptr,
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

        filter_pool->unlock_buffers();
        return *buf_size;

    } else {
        /* Compressing raw data into jpegls-encoding */

        jpegls::span<uint8_t> raw_data{reinterpret_cast<uint8_t*>(*buf), *buf_size};
        const auto out_buf = jpegls::encode(raw_data, config);
        *buf = out_buf.data;
        *buf_size = out_buf.size;

        return out_buf.size;
    }
}

VISIBLE
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
        unsigned int nblocks = (ndims == 1) ? 1 : std::accumulate(
                chunkdims, chunkdims + ndims - 1, 1, std::multiplies<int>());

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

VISIBLE
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

VISIBLE
H5PL_type_t H5PLget_plugin_type() {  // NOLINT
    return H5PL_TYPE_FILTER;
}

VISIBLE
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
