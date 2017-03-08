#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<iostream>
#include<sstream>
#include<fstream>
#include<vector>
#include<cmath>
#include<ctime>
#include<cstdlib>
#include <unistd.h>
using namespace cv;
using namespace std;

#define numG 3

float tau = 2.5;
float bthreshold = 0.75;
float alpha = 0.1;
float ivarince = 36.0;
int _resize = 0;

struct spm{           // single pixel model
  vector<Vec3b> means;
  vector<float> variances;
  vector<float> pi;
  vector<int> matched;
  spm()
  {
    means = vector<Vec3b>(numG);
    variances = vector<float> (numG,ivarince);
    pi = vector<float> (numG,1/3.0);
    matched = vector<int>(numG);
  }
};

struct model{
  vector< vector<spm> > GMMat;
  model(int nrows, int ncols)
  {
    GMMat = vector< vector<spm> > (nrows, vector<spm> (ncols));
  }
};

bool checkMatch(Vec3b & pixVal, spm & pmodel, int gaussNum)
{
  return norm(pixVal,pmodel.means[gaussNum]) <= tau*sqrt(pmodel.variances[gaussNum]);
}

int main(int argc, char** argv)  // 1) filename 2) alpha 3) bthreshold
{
  VideoCapture video(argv[1]);
  alpha = atof(argv[2]);
  bthreshold = atof(argv[3]);
  int width,height;

  if(argc == 5)
  {
    _resize = atoi(argv[4]);
  }

  if(_resize == 1)
  {
    video.set(CV_CAP_PROP_FRAME_WIDTH, 240);
    video.set(CV_CAP_PROP_FRAME_HEIGHT, 120);
    width = 240;
    height = 120;
  }
  else
  {
    width = video.get(CV_CAP_PROP_FRAME_WIDTH);
    height = video.get(CV_CAP_PROP_FRAME_HEIGHT);
  }

  Size S = Size(width,height);
  int ex = static_cast<int>(video.get(CV_CAP_PROP_FOURCC));
  VideoWriter BackoutputVideo;
  VideoWriter ForeOutputVideo;
  string BACK_NAME = "background/alpha"+to_string(alpha)+"thr"+to_string(bthreshold)+"tau"+to_string(tau)+".mp4";
  string FORE_NAME = "foreground/alpha"+to_string(alpha)+"thr"+to_string(bthreshold)+"tau"+to_string(tau)+".mp4";
  BackoutputVideo.open(BACK_NAME, ex, video.get(CV_CAP_PROP_FPS), S, true);
  ForeOutputVideo.open(FORE_NAME, ex, video.get(CV_CAP_PROP_FPS), S, true);
  Mat frame;
  Mat fore_image(height,width,CV_8UC3);
  Mat back_image(height,width,CV_8UC3);
  model currentModel(height,width);
  vector <bool> matchStatus(numG,false);
  vector<pair<float,int> > sorted_weights(numG);
  namedWindow("foreground",WINDOW_AUTOSIZE);
  namedWindow("background",WINDOW_AUTOSIZE);
  for(;;)
  {
    video >> frame;
    if(frame.empty())
    {
      break;
    }
    if(_resize == 1)
    {
      resize(frame, frame, Size(240, 120), 0, 0, INTER_CUBIC);
    }
    for (int rowid = 0; rowid < height; rowid++)
    {
      for(int colid = 0; colid < width; colid++)
      {
        spm & pmodel = currentModel.GMMat[rowid][colid];
        Vec3b pixVal = frame.at<Vec3b>(rowid,colid);
        bool flag = false;
        for(int i = 0; i < numG; i++)
        {
          if(checkMatch(pixVal,pmodel,i))
          {
            matchStatus[i] = true;
            flag = true;
          }
          else
          {
            matchStatus[i] = false;
          }
        }
        float sumpi = 0.0;
        if(flag)
        {
          for(int i =0; i < numG; i++)
          {
            if(matchStatus[i])
            {
              float rho = alpha*pow(2*3.141*pmodel.variances[i],-numG/2)*(exp(-0.5*pow(norm(pixVal,pmodel.means[i]),2)/pmodel.variances[i]));
              pmodel.variances[i] = (1-rho)*pmodel.variances[i] + rho*pow(norm(pixVal,pmodel.means[i]),2);
              pmodel.means[i] = (1-rho)*pmodel.means[i] + rho*pixVal;
              pmodel.pi[i] = (1-alpha)*pmodel.pi[i] + alpha;
              sumpi += pmodel.pi[i];
              pmodel.matched[i] += 1;
            }
            else
            {
              pmodel.pi[i] = (1-alpha)*pmodel.pi[i];
              sumpi += pmodel.pi[i];
            }
          }
        }

        else
        {
          int worst_index = 0;
          float worst_val = pmodel.pi[0]/sqrt(pmodel.variances[0]);
          for(int i = 1; i < numG; i++)
          {
            float temp = pmodel.pi[i]/sqrt(pmodel.variances[i]);
            if(temp < worst_val)
            {
              worst_val = temp;
              worst_index = i;
            }
          }
          pmodel.means[worst_index] = pixVal;
          pmodel.variances[worst_index] = ivarince;
          sumpi = 1 - pmodel.pi[worst_index] + 0.001;
          pmodel.pi[worst_index] = 0.001;
          pmodel.matched[worst_index] = 1;
        }
        for(int i = 0 ; i < numG; i++)
        {
          pmodel.pi[i] /= sumpi;
          sorted_weights[i] = {pmodel.pi[i],i};
        }
        sort(sorted_weights.rbegin(),sorted_weights.rend());
        float running_weight = 0.0;
        bool isForeground = true;
        for(int i = 0; i < numG; i++)
        {
          if(running_weight > bthreshold)
            break;
          running_weight += sorted_weights[i].first;
          if(checkMatch(pixVal,pmodel,sorted_weights[i].second))
          {
            isForeground = false;
            break;
          }
        }
        back_image.at<Vec3b>(rowid,colid) = pmodel.means[sorted_weights[0].second];
        if(isForeground)
        {
          fore_image.at<Vec3b>(rowid, colid)[2] = 255;
        }
        else
        {
          fore_image.at<Vec3b>(rowid, colid)[2] = 0;
        }
      }
    }
    imshow("foreground", fore_image);
    imshow("background", back_image);
    BackoutputVideo << back_image;
    ForeOutputVideo << fore_image;
    waitKey(1);
  }
}
