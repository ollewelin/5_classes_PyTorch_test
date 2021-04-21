#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "model.h"

const int input_img_size = 224;

cv::Mat makeSquareImg(cv::Mat frame, int outputImageSize)
{
  cv::Mat rectImage, graymat;
  frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
  //Resize input image to fit into a example 224 x 224 dataset images
  //First step scale the input image so the least side h or w will be resized to example 224
  //Next step crop the larger h or w to example 224 pixel as well, crop with images in center
  //bool inp_landscape;
  int x_start = 0;
  int y_start = 0;
  int inputSize = 0;
  //Find out the smallest side hight or width of the input image
  if (frame.rows == 0 || frame.cols == 0)
  {
    //Zero divition protection here.
    printf("Error! Zero divition protection input image rows = %d cols = %d. Exit program.\n", frame.rows, frame.cols);
    exit(0);
  }
  else
  {
    if (frame.rows > frame.cols)
    {
      //Input images is a portrait mode
      //Calculate the starting point of the square rectangle
      inputSize = frame.cols;
      y_start = (frame.rows / 2) - inputSize / 2;
      x_start = 0;
    }
    else
    {
      //Input images is a landscape mode
      //Make a square rectangle of input image
      inputSize = frame.rows;
      x_start = (frame.cols / 2) - inputSize / 2;
      y_start = 0;
    }
    //Make a square rectangle of input image
    //Mat rect_part(image, Rect(rand_x_start, rand_y_start, Width, Height));//Pick a small part of image
    cv::Mat rectImageTemp(frame, cv::Rect(x_start, y_start, inputSize, inputSize)); //
    //Size size(input_image_width,input_image_height);//the dst image size,e.g.100x100
    cv::Size outRectSize(outputImageSize, outputImageSize);
    //resize(src,dst,size);//resize image
    cv::resize(rectImageTemp, rectImage, outRectSize);
  }
  return rectImage;
}

int main(int arc, char **argv)
{
  std::string loc = argv[1];
  //Prepare for GPU
  torch::DeviceType device_type;
  srand(time(NULL));
  if (torch::cuda::is_available())
  {
    std::cout << "CUDA available! Classify test image on GPU." << std::endl;
    device_type = torch::kCUDA;
  }
  else
  {
    std::cout << "Classify test image on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  // Load image with OpenCV.
  cv::Mat img = cv::imread(loc);
  cv::Mat imgR = makeSquareImg(img, input_img_size);
  // Convert the image and label to a tensor.
  torch::Tensor img_tensor = torch::from_blob(imgR.data, {1, imgR.rows, imgR.cols, 3}, torch::kF32);
  img_tensor = img_tensor.permute({0, 3, 1, 2}); // convert to CxHxW
                                                 //  img_tensor = img_tensor.to(device).contiguous();
  img_tensor = img_tensor.to(device);

  cv::Mat testImg(input_img_size, input_img_size, CV_32FC3);
  torch::Tensor CPUtensor;
  torch::Tensor CPUtens = img_tensor.to(torch::kCPU);
  float *index_ptr_testImg = testImg.ptr<float>(0);
  for (int r = 0; r < testImg.rows; r++)
  {
    for (int c = 0; c < testImg.cols; c++)
    {
      for (int rgb = 0; rgb < testImg.channels(); rgb++)
      {
        //*index_ptr_testImg = CPUtens[kTestBatchSize-1][rgb][r][c].item<float>();
        *index_ptr_testImg = CPUtens[0][rgb][r][c].item<float>();
        index_ptr_testImg++;
      }
    }
  }
  cv::imshow("dataset img", testImg);
  // Load the model.
  ObscureResNet model(5 /*nr of classes*/);
  torch::load(model, "./latest_model.pt");
  model->to(device);

  // Predict the probabilities for the classes.
  torch::Tensor log_prob = model(img_tensor);
  torch::Tensor prob = torch::exp(log_prob);

  int i = 0;
  printf("Probability of being\n\
    class0 = %.2f percent\n\
    class1 = %.2f percent\n\
    class2 = %.2f percent\n\
    class3 = %.2f percent\n\    
    class4 = %.2f percent\n",
         prob[i][0].item<float>() * 100., prob[i][1].item<float>() * 100., prob[i][2].item<float>() * 100., prob[i][3].item<float>() * 100., prob[i][4].item<float>() * 100.);
  cv::waitKey(5000);
  return 0;
}
