#include <stdio.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

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

std::unique_ptr<YoussefInkDetection> mnist_;

struct InputImage
{
    static constexpr const int width = 32; // pixels
    static constexpr const int height = 32; // pixels
    static constexpr const int channels = 1; // intensity
    size_t strideBytes = channels * width * sizeof(float);
    int pixelCount = width * height;
    std::array<float, width * height * channels> pixels{}; // color data
};

// Clip to x1, y1, w1, h1
InputImage ReadImageSliceToInputImage(std::string filename, int x1, int y1, int w1, int h1)
{
    InputImage image;

    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    if (tif) {
        uint32_t w, h;
        size_t npixels;
        uint32_t* raster;

        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
        npixels = w * h;
        raster = (uint32_t*)_TIFFmalloc(npixels * sizeof(uint32_t));
        if (raster != NULL) {
            if (TIFFReadRGBAImage(tif, w, h, raster, 0)) {
                for (int y = y1; y < h && y < h1; y++)
                {
                    for (int x = x1; x < w && x < w1; x++)
                    {
                        int offset = y * w + x;
                        uint32_t* pixel = raster + offset;
                        uint8_t a = (*pixel) >> 24;
                        uint8_t b = (*pixel) >> 16;
                        uint8_t g = (*pixel) >> 8;
                        uint8_t r = (*pixel) & 0x000000FF;

                        int offset2 = (y - y1) * image.width + (x - x1);
                        if (offset2 < image.pixelCount)
                        {
                            float intensity = r / (float)UINT8_MAX;
                            image.pixels[offset2] = intensity;
                        }
                    }
                }
            }
            _TIFFfree(raster);
        }
        TIFFClose(tif);
    }

    return image;
}

void FillInput(InputImage& image)
{
    float* output = mnist_->input_image_.data();

    std::fill(mnist_->input_image_.begin(), mnist_->input_image_.end(), 0.f);

    for (unsigned y = 0; y < YoussefInkDetection::height_; y++) {
        for (unsigned x = 0; x < YoussefInkDetection::width_; x++) {
            int offset = y * image.width + x;
            output[x] += image.pixels[offset];
        }
        output += YoussefInkDetection::width_;
    }
}

struct OutputImage
{
    static constexpr const int width = 8; // pixels
    static constexpr const int height = 8; // pixels
    static constexpr const int channels = 3; // rgb
    size_t strideBytes = channels * width * sizeof(uint8_t);
    int pixelCount = width * height;
    std::array<uint8_t, width * height * channels> pixels{}; // color data
};

OutputImage ConvertResultToOutputImage()
{
    OutputImage image;

    float* results = mnist_->results_.data();
    for (unsigned y = 0; y < image.height; y++) {
        for (unsigned x = 0; x < image.width; x++) {
            float result = results[x];
            uint8_t r = (uint8_t)roundf(result * UINT8_MAX);
            int pixelOffset = y * image.height + x;
            image.pixels[pixelOffset * image.channels + 0] = r;
            image.pixels[pixelOffset * image.channels + 1] = r;
            image.pixels[pixelOffset * image.channels + 2] = r;
        }
        results += image.width;
    }

    return image;
}

void WriteResultToImage(std::string filename, OutputImage& image)
{
    int ret = stbi_write_png(filename.c_str(), image.width, image.height, image.channels, image.pixels.data(), image.strideBytes);
    assert(ret > 0); // success.
}

int main()
{
	printf("hello world");
    try {
        mnist_ = std::make_unique<YoussefInkDetection>();
    }
    catch (const Ort::Exception& exception) {
        //MessageBoxA(nullptr, exception.what(), "Error:", MB_OK);
        return 0;
    }

    // https://vesuvius.virtual-void.net/scroll/1/segment/20230827161847/#u=2524&v=4582&zoom=0.046&rot=90&flip=false&layer=34

    int x = 0;
    int y = 0;
    int sliceWidth = 5048;
    int sliceHeight = 9163;

    int tileWidth = 32;
    int tileHeight = 32;

    for (int y = 0; y < sliceWidth; y+=tileHeight)
    {
        for (int x = 0; x < sliceWidth; x+=tileWidth)
        {
            // Todo(kbostelmann): Read layer 15 until 45 (+30 slices). Fill into inputs.
            auto inputImage = ReadImageSliceToInputImage("20230827161847/layers/15.tif", x, y, x + tileWidth, y + tileHeight);
            FillInput(inputImage);

            mnist_->Run();

            auto outputImage = ConvertResultToOutputImage();
            // Todo(kbostelmann): bilinear interpolate image resolution to 32x32.

            std::string filename = "out_" + std::to_string(x) + "_" + std::to_string(y) + ".png";
            WriteResultToImage(filename, outputImage);

            printf(filename.c_str());
        }
    }
}