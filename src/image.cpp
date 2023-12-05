#include <memory>
#include <string>

#include <filesystem>
namespace fs = std::filesystem;

#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <tiff.h>
#include <tiffio.h>

std::unique_ptr<Image> ReadImageTIFF(std::string filename)
{
    std::unique_ptr<Image> image = nullptr;

    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    if (tif) {
        uint32_t w, h;
        size_t npixels;
        uint32_t* raster;

        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        npixels = w * h;
        raster = (uint32_t*)_TIFFmalloc(npixels * sizeof(uint32_t));
        image = std::make_unique<Image>(w, h);
        if (raster != NULL) {
            if (TIFFReadRGBAImage(tif, w, h, raster, 0)) {
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int offset = (h - 1 - y) * w + x; // Invert y-axis.
                        uint32_t* pixel = raster + offset;
                        uint8_t a = (*pixel) >> 24;
                        uint8_t b = (*pixel) >> 16;
                        uint8_t g = (*pixel) >> 8;
                        uint8_t r = (*pixel) & 0x000000FF;

                        int pixelOffset = y * image->width + x;
                        image->pixels[pixelOffset * image->channels + 0] = r;
                        image->pixels[pixelOffset * image->channels + 1] = g;
                        image->pixels[pixelOffset * image->channels + 2] = b;
                    }
                }
            }
            _TIFFfree(raster);
        }
        TIFFClose(tif);
    }

    return image;
}

void CopyToInput(Image& image, int x0, int y0, float* input, int inputWidth, int inputHeight)
{
    for (int y = y0; y < y0 + inputWidth; y++)
    {
        for (int x = x0; x < x0 + inputHeight; x++)
        {
            uint8_t gray = image.GetPixelGray(x, y);
            float intensity = gray / (float)UINT8_MAX;

            int pixelOffset = (y - y0) * inputWidth + (x - x0);
            input[pixelOffset] = intensity;
        }
    }
}

ResultImage ConvertResultToImage(float* results)
{
    ResultImage image;

    for (unsigned y = 0; y < image.height; y++) {
        for (unsigned x = 0; x < image.width; x++) {
            float result = results[x];
            uint8_t gray = (uint8_t)roundf(result * UINT8_MAX);
            int pixelOffset = y * image.width + x;
            image.pixels[pixelOffset * image.channels + 0] = gray;
            image.pixels[pixelOffset * image.channels + 1] = gray;
            image.pixels[pixelOffset * image.channels + 2] = gray;
        }
        results += image.width;
    }

    return image;
}

std::unique_ptr<Image> ReadImage(std::string filename)
{
    if (fs::path(filename).extension() == ".tif") {
        return ReadImageTIFF(filename);
    }
    // else fallback to other formats.

    int width;
    int height;
    int channels;

    unsigned char* imgData = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (imgData != NULL) {
        std::unique_ptr<Image> image = std::make_unique<Image>(width, height, channels);
        memcpy(image->pixels, imgData, image->sizeBytes());
        stbi_image_free(imgData);
        return image;
    }
    else {
        return nullptr; // failed to load image.
    }
}

void WriteImagePNG(std::string filename, Image& image)
{
    int ret = stbi_write_png(filename.c_str(), image.width, image.height, image.channels, image.pixels, image.strideBytes());
    assert(ret > 0); // success.
}

void WriteImage(std::string filename, Image& image)
{
    if (fs::path(filename).extension() == ".png") {
        return WriteImagePNG(filename, image);
    }
    // else write other formats.
    assert(false); // not implemented.
}

void CopyToImage(Image& src, Image& dst, int x0, int y0)
{
    for (int y = y0; y < y0 + src.height; y++)
    {
        for (int x = x0; x < x0 + src.width; x++)
        {
            int srcPixelOffset = (y - y0) * src.width + (x - x0);
            int dstPixelOffset = y * dst.width + x;

            dst.pixels[dstPixelOffset * dst.channels + 0] = src.pixels[srcPixelOffset * src.channels + 0];
            dst.pixels[dstPixelOffset * dst.channels + 1] = src.pixels[srcPixelOffset * src.channels + 1];
            dst.pixels[dstPixelOffset * dst.channels + 2] = src.pixels[srcPixelOffset * src.channels + 2];
        }
    }
}

void FillPixels(ResultImage& image, uint8_t red, uint8_t blue, uint8_t green)
{
    for (unsigned y = 0; y < image.height; y++) {
        for (unsigned x = 0; x < image.width; x++) {
            int pixelOffset = y * image.width + x;
            image.pixels[pixelOffset * image.channels + 0] = red;
            image.pixels[pixelOffset * image.channels + 1] = blue;
            image.pixels[pixelOffset * image.channels + 2] = green;
        }
    }
}

std::unique_ptr<Image> ResizeImage(ResultImage& image, int width, int height)
{
    std::unique_ptr<Image> result = std::make_unique<Image>(width, height);
    stbir_resize_uint8_linear(image.pixels.data(), image.width, image.height, image.strideBytes,
        result->pixels, result->width, result->height, result->strideBytes(),
        (stbir_pixel_layout)3);
    return result;
}