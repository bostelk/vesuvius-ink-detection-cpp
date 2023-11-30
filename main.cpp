#include <stdio.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>

#include <filesystem>
namespace fs = std::filesystem;

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <tiff.h>
#include <tiffio.h>

template <typename T>
static void softmax(T& input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp(input[i] - rowmax);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

static float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

template <typename T>
static void sigmoid(T& input) {
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = sigmoid(input[i]);
    }
}


// This is the structure to interface with the ink detection model
// After instantiation, set the input_image_ data to be the 28x28 pixel image of the number to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
struct YoussefInkDetection {
    YoussefInkDetection() {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(),
            input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(),
            output_shape_.data(), output_shape_.size());
    }

    std::ptrdiff_t Run() {
        const char* input_names[] = { "arg0" };
        const char* output_names[] = { "decoder_1" };

        Ort::RunOptions run_options;
        //Ort::GetApi().RunOptionsSetRunLogSeverityLevel(run_options, 0); VERBOSE
        session_.Run(run_options, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
        sigmoid(results_);
        // There is no single "most confident" label to return.
        return result_;
    }

    static constexpr const int width_ = 32;
    static constexpr const int height_ = 32;

    std::array<float, width_* height_> input_image_{};
    std::array<float, 8 * 8> results_{};
    int64_t result_{ 0 };

private:
    Ort::Env env;
    Ort::Session session_{ env, L"vesuvius-ink-detection.onnx", Ort::SessionOptions{nullptr} };

    Ort::Value input_tensor_{ nullptr };
    std::array<int64_t, 4> input_shape_{ 1, 1, width_, height_ };

    Ort::Value output_tensor_{ nullptr };
    std::array<int64_t, 4> output_shape_{ 1, 1, 8, 8 };
};

std::unique_ptr<YoussefInkDetection> inkDetection_;

// A window of a slice to detect ink within.
struct InputImage
{
    static constexpr const int width = 32; // pixels
    static constexpr const int height = 32; // pixels
    static constexpr const int channels = 1; // intensity
    size_t strideBytes = channels * width * sizeof(float);
    int pixelCount = width * height;
    std::array<float, width * height * channels> pixels{}; // color data
};


// A generic image. Allocates heap memory on construction.
struct Image
{
    int width; // pixels
    int height; // pixels
    int channels; // rgb
    size_t strideBytes() {
        return channels * width * sizeof(uint8_t);
    }
    size_t sizeBytes() {
        return height * strideBytes();
    }
    int pixelCount() {
        return width * height;
    }
    uint8_t* pixels; // color data
    uint8_t GetPixelGray(int x, int y) {
        int pixelOffset = y * width + x;
        return pixels[pixelOffset * channels + 0];
    }
    Image(int width_, int height_) : width(width_), height(height_), channels(3) {
        pixels = (uint8_t*)malloc(sizeBytes());
        memset(pixels, 0, sizeBytes());
    }
    Image(int width_, int height_, int channels_) : width(width_), height(height_), channels(channels_) {
        pixels = (uint8_t*)malloc(sizeBytes());
        memset(pixels, 0, sizeBytes());
    }
    ~Image() {
        free(pixels);
    }
};

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

void CopyToInput(Image& image, int x0, int y0)
{
    float* output = inkDetection_->input_image_.data();

    std::fill(inkDetection_->input_image_.begin(), inkDetection_->input_image_.end(), 0.f);

    for (int y = y0; y < y0 + YoussefInkDetection::height_; y++)
    {
        for (int x = x0; x < x0 + YoussefInkDetection::width_; x++)
        {
            uint8_t gray = image.GetPixelGray(x, y);
            float intensity = gray / (float)UINT8_MAX;

            int pixelOffset = (y - y0) * YoussefInkDetection::width_ + (x - x0);
            output[pixelOffset] = intensity;
        }
    }
}

// An ink detection result converted to an image.
struct ResultImage
{
    static constexpr const int width = 8; // pixels
    static constexpr const int height = 8; // pixels
    static constexpr const int channels = 3; // rgb
    size_t strideBytes = channels * width * sizeof(uint8_t);
    int pixelCount = width * height;
    std::array<uint8_t, width * height * channels> pixels{}; // color data
};

ResultImage ConvertResultToImage()
{
    ResultImage image;

    float* results = inkDetection_->results_.data();
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
    } else {
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

// Copy a source image into a destination image region located at (x0, y0).
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

int main()
{
	printf("hello world");
    try {
        inkDetection_ = std::make_unique<YoussefInkDetection>();
    }
    catch (const Ort::Exception& exception) {
        //MessageBoxA(nullptr, exception.what(), "Error:", MB_OK);
        return 0;
    }

    // https://vesuvius.virtual-void.net/scroll/1/segment/20230827161847/#u=2524&v=4582&zoom=0.046&rot=90&flip=false&layer=34

    int x0 = 1248;
    int y0 = 9163 / 2;
    int sliceWidth = 5048;
    int sliceHeight = 9163;

    Image image(sliceWidth, sliceHeight); // Allocates.

    for (int y = y0; y < sliceWidth; y+= YoussefInkDetection::width_)
    {
        for (int x = x0; x < sliceWidth; x+= YoussefInkDetection::height_)
        {
            printf((std::to_string(x) + " " + std::to_string(y) + "\n").c_str());

            auto mask = ReadImage("20230827161847/20230827161847_mask.png");
            if (!mask || mask->GetPixelGray(x, y) != 255)
            {
                continue; // Skip because pixel is not on papyrus.
            }

            // Todo(kbostelmann): Read layer x until y (+30 slices). Fill into inputs.
            int initialLayerIndex = 15;
            for (int layerIndex = initialLayerIndex; layerIndex < initialLayerIndex + 30; layerIndex++)
            {
                auto layer = ReadImage("20230827161847/layers/" + std::to_string(layerIndex) + ".tif");
                if (layer) {
                    CopyToInput(*layer, x, y);
                }
            }

            inkDetection_->Run();

            auto imageTile = ConvertResultToImage();

            // Resize to input resolution.
            auto imageTileResized = ResizeImage(imageTile, YoussefInkDetection::width_, YoussefInkDetection::height_);

            CopyToImage(*imageTileResized, image, x, y);
            std::string filename = "out.png";
            WriteImage(filename, image);
        }
    }
}