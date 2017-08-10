/*
 * =====================================================================================
 *
 *       Filename:  ProbabilityMapping.cc
 *
 *    Description:
 *
 *        Version:  0.1
 *        Created:  01/21/2016 10:39:12 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Josh Tang, Rebecca Frederick
 *
 *        version: 1.0
 *        created: 8/9/2016
 *        Log: fix a lot of bug, Almost rewrite the code.
 *
 *        author: He Yijia
 *
 * =====================================================================================
 */

#include <cmath>
#include <opencv2/opencv.hpp>
#include <numeric>
#include "ProbabilityMapping.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "ORBmatcher.h"
#include "LocalMapping.h"
#include <stdint.h>
#include <stdio.h>
#include<fstream>
//#include<octomap/octomap.h>
//#include<octomap/OcTree.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include<pcl/point_types.h>
#include<pcl/visualization/cloud_viewer.h>

#include<iostream>

#define DEBUG 1
#define _USE_MATH_DEFINES
//#define InterKeyFrameChecking


void saveMatToCsv(cv::Mat data, std::string filename)
{
    std::ofstream outputFile(filename.c_str());
   outputFile<< cv::format(data,"CSV")<<std::endl;
    outputFile.close();
}

template<typename T>
float bilinear(const cv::Mat& img, const float& y, const float& x)
{

    int x0 = (int)std::floor(x);
    int y0 = (int )std::floor(y);
    int x1 = x0 + 1;
    int y1 =  y0 + 1;

    float x0_weight = x1 - x;
    float y0_weight = y1 - y;
    float x1_weight = 1.0f - x0_weight;
    float y1_weight = 1.0f - y0_weight;

    float interpolated =
            img.at<T>(y0 , x0 ) * x0_weight*y0_weight + img.at<T>(y0 , x1)* x1_weight*y0_weight +
           img.at<T>(y1 , x0 ) * x0_weight *y1_weight+ img.at<T>(y1 , x1)* x1_weight*y1_weight ;


return interpolated;
}

float ProbabilityMapping::bilinear1(const cv::Mat& img, const float& y, const float& x)
{
    int x0 = (int)std::floor(x);
    int y0 = (int )std::floor(y);
    int x1 = x0 + 1;
    int y1 =  y0 + 1;

    float x0_weight = x1 - x;
    float y0_weight = y1 - y;
    float x1_weight = 1.0f - x0_weight;
    float y1_weight = 1.0f - y0_weight;
   uchar* d=&img.data[y0*img.step+x0];

  return (x0_weight*y0_weight*float(d[0])+x1_weight*y0_weight*float(d[1])+x0_weight*y1_weight*float(d[img.step])+x1_weight*y1_weight*float(d[img.step+1]))/255;
}

ProbabilityMapping::ProbabilityMapping(ORB_SLAM2::Map* pMap):mpMap(pMap)
{
 mbFinishRequested = false; //init
}

void ProbabilityMapping::Run()
{
    while(1)
    {
        if(CheckFinish()) break;
        sleep(1);
       // TestSemiDenseViewer();
       SemiDenseLoop();

    }
}

void ProbabilityMapping::TestSemiDenseViewer()
{
        unique_lock<mutex> lock(mMutexSemiDense);
        vector<ORB_SLAM2::KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
        if(vpKFs.size() < 2)
        {
            return;
        }
        for(size_t i =0;i < vpKFs.size(); i++ )
        {
            ORB_SLAM2::KeyFrame* pKF = vpKFs[i];
            if(pKF->isBad() || pKF->semidense_flag_)
                continue;

            cv::Mat image = pKF->GetImage();
            std::vector<std::vector<depthHo> > temp_ho (image.rows, std::vector<depthHo>(image.cols, depthHo()) );

            for(int y = 0; y < image.rows; ){
              for(int x = 0; x < image.cols; ){

                       depthHo dh;
                       dh.depth = 100.0;   // const
                       float X = dh.depth*(x- pKF->cx ) / pKF->fx;
                       float Y = dh.depth*(y- pKF->cy ) / pKF->fy;
                       cv::Mat Pc = (cv::Mat_<float>(4,1) << X, Y , 1/dh.depth, 1); // point in camera frame.
                       cv::Mat Twc = pKF->GetPoseInverse();
                       cv::Mat pos = Twc * Pc;
                       dh.Pw<< pos.at<float>(0),pos.at<float>(1),pos.at<float>(2);
                       dh.supported = true;
                       temp_ho[y][x] = dh;  // save point to keyframe semidense map
                       pKF->SemiDensePointSets_.at<float>(y,3*x+0) = pos.at<float>(0);
                       pKF->SemiDensePointSets_.at<float>(y,3*x+1) = pos.at<float>(1);
                       pKF->SemiDensePointSets_.at<float>(y,3*x+2) = pos.at<float>(2);
                         x = x+4; // don't use all pixel to test
              }
              y = y+4;
            }

            pKF->SemiDenseMatrix = temp_ho;
            pKF->semidense_flag_ = true;
        }
        cout<<"semidense_Info:    vpKFs.size()--> "<<vpKFs.size()<<std::endl;


}

void ProbabilityMapping::SemiDenseLoop(){

  unique_lock<mutex> lock(mMutexSemiDense);

  vector<ORB_SLAM2::KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
  cout<<"semidense_Info:    vpKFs.size()--> "<<vpKFs.size()<<std::endl;


  if(vpKFs.size() < covisN+3){return;}

  for(size_t i =0;i < vpKFs.size(); i++ )
  {
      ORB_SLAM2::KeyFrame* kf = vpKFs[i];
      if(kf->isBad() || kf->semidense_flag_||kf->semidense_begin_inter)
        continue;
        std::vector<ORB_SLAM2::KeyFrame*> closestMatches = kf->GetBestCovisibilityKeyFrames(covisN);
        if(closestMatches.size() < covisN) {continue;};

        float max_invdepth=0;
        float min_invdepth=0;
        // get max_dephth  and min_depth in current key frame to limit search range
        StereoSearchConstraints(kf, &min_invdepth, &max_invdepth);
        //cout<<"(min_invdepth,max_invdepth)="<<min_invdepth<<","<<max_invdepth<<endl;
        if((max_invdepth-min_invdepth)<0){cout<<"woory"<<endl;}

   //qwe
   cv::Mat image = kf->GetImage();
        //cv::Mat image_debug = image.clone();

       // std::vector<std::vector<depthHo> > temp_ho (image.rows, std::vector<depthHo>(image.cols, depthHo()) );

        std::vector <cv::Mat> F;
        vector<float> vec_cos;
        vector<cv::Mat> vec_t21;
        F.clear();
        for(size_t j=0; j<closestMatches.size(); j++)
        {
          ORB_SLAM2::KeyFrame* kf2 = closestMatches[ j ];
          cv::Mat F12 = ComputeFundamental(kf,kf2);
          F.push_back(F12);
          cv::Mat R1w_ = kf->GetRotation();
          cv::Mat R2w_ = kf2->GetRotation();

          cv::Mat t1w = kf->GetTranslation();

          cv::Mat t2w = kf2->GetTranslation();
          cv::Mat r21=R2w_*R1w_.t();
          cv:: Mat pjj_t21=-R2w_*R1w_.t()*t1w+t2w;
          vec_t21.push_back(pjj_t21);
          cv::Mat pjj_x=(cv::Mat_<float>(3,1)<<1,1,1);
          cv::Mat pjj_y=r21*pjj_x;
          cv::Mat cosmat=pjj_x.t()*pjj_y;
          float cos=cosmat.at<float>(0,0)/(sqrt(3)*sqrt(pow(pjj_y.at<float>(0,0),2)+pow(pjj_y.at<float>(1,0),2)+pow(pjj_y.at<float>(2,0),2)));

         //cout<<"jiaodu="<<cos<<endl;
          vec_cos.push_back(cos);
        }

/*******************pippeijieguo*************
                for(size_t j=0;j<closestMatches.size();j++){
            //if(j!=closestMatches.size()-1){continue;}
                     std::vector<pair<float,float >> point1,point2,point3;
                     std::vector<float> ap,bp,cp;
                    cv::Mat pjjimage=image.clone();
                     ORB_SLAM2::KeyFrame* kf2 = closestMatches[j];
                    cv::Mat pjjimagepjj1=kf2->im_;
                    cv::Mat pjjimage1=pjjimagepjj1.clone();
                    cv::Mat pjjimage2=pjjimagepjj1.clone();
                    cv::Mat F12 = F[j];
                   float cos=vec_cos[j];
                  if(cos>0.998){continue;}
                   cv::Mat R1w = kf->GetRotation();
                   cv::Mat t1w = kf->GetTranslation();
                   cv::Mat R2w = kf2->GetRotation();
                   cv::Mat t2w = kf2->GetTranslation();
                 //   for(size_t p=0;p<kf->GradImg.rows;p++){
                   //     for(size_t j=0;j<kf->GradImg.cols;j++){
                      //      if(kf->GradImg.at<float>(p,j)<15){
                          //      kf->GradImg.at<float>(p,j)=0;
                          //      kf->GradTheta.at<float>(p,j)=0;
                       //    }
                       //     if(kf2->GradImg.at<float>(p,j)<15){
                      //          kf2->GradImg.at<float>(p,j)=0;
                     //             kf2->GradTheta.at<float>(p,j)=0;
                    //        }
                 //       }
                 //   }
                 //   cv::imshow("gradient",kf->GradImg);
                //    cv::imshow("gradtheta",kf->GradTheta);
                //    cv::imshow("gradient2",kf2->GradImg);
                //    cv::imshow("gradtheta2",kf2->GradTheta);
                //    cv::waitKey(0);

                    //cv::Mat R12 = R1w*R2w.t();
                   // cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
                   // cv::Mat R21=R2w*R1w.t();
                    //float duijiao=R21.at<float>(0,0)+R21.at<float>(1,1)+R21.at<float>(2,2);
                    cout<<"cos="<<cos<<endl;
                   cv::Mat t21=-R2w*R1w.t()*t1w+t2w;
                    //cout<<"kf->mnid="<<kf->mnId<<endl;
                    //cout<<"kf2->mnid="<<kf2->mnId<<endl;
                  //  cout<<"R21="<<R21<<endl;
                    for(int y=2;y<image.rows-2;y++){
                        for(int x=2;x<image.cols-2;x++){
                            if(kf->GradImg.at<float>(y,x) < lambdaG){continue;}
                            float pixel =(float) image.at<uchar>(y,x); //maybe it should be cv::Mat
                            float best_u(0.0),best_v(0.0),pjju(0.0),pjjv(0.0);
                            depthHo dh;
                           // cv::Mat A;
                          // getWarpMatrixAffine(kf,kf2,x,y,1, 5,A);
                            float a=0.0,b=0.0,c=0.0;
                            vector<int> vec_u;
                              std::vector<pair<int,int >> point4;
                            vector<depthHo> vec_depth;

           //            EpipolarSearch(kf, kf2, x, y, pixel, min_invdepth, max_invdepth, &dh,F12,pjju,pjjv,best_u,best_v,kf->GradTheta.at<float>(y,x),a,b,c);
                // EpipolarSearch_h(kf, kf2, x, y, pixel, min_invdepth, max_invdepth, &dh,F12,pjju,pjjv,best_u,best_v,kf->GradTheta.at<float>(y,x),a,b,c);
                            EpipolarSearch_after(kf, kf2, x, y, pixel, min_invdepth, max_invdepth, &dh,F12,pjju,pjjv,best_u,best_v,kf->GradTheta.at<float>(y,x),a,b,c);
                        if(dh.supported){
                          //  for(size_t i=0;i<vec_u.size();i++){
                          //      float v1=-(a/b*vec_u[i]+c/b);
                          //    point4.push_back(make_pair(v1,vec_u[i]));
                            //   cv::Point uv1(vec_u[i],v1);
                            //    cv::circle(pjjimage1,uv1,2,cv::Scalar(255,255,255),1,8,0);
                          //  }

                          //  cv::Point xy(x,y);
                          // cv::Point best_uv(best_u,best_v);
                        //   cv::Point uv(pjju,pjjv);
                           point1.push_back(make_pair(y,x));
                           point2.push_back(make_pair(best_v,best_u));
                           point3.push_back(make_pair(pjjv,pjju));
                           ap.push_back(a);
                           bp.push_back(b);
                            cp.push_back(c);
                           //cv::line(pjjimage1,cv::Point(0,-(c/b)),cv::Point(pjjimage1.cols,-((a/b)*pjjimage1.cols+c/b)),cv::Scalar(255,255,255),1,8,0);
                       //    cv::line(pjjimage2,cv::Point(0,-(c/b)),cv::Point(pjjimage1.cols,-((a/b)*pjjimage1.cols+c/b)),cv::Scalar(255,255,255),1,8,0);
                        //     cv::circle(pjjimage,xy,5,cv::Scalar(255,255,255),1,8,0);
                           //    cv::circle(pjjimage1,uv,5,cv::Scalar(255,255,255),1,8,0);
                              //    cv::circle(pjjimage2,best_uv,5,cv::Scalar(255,255,255),1,8,0);
                                //  cv::imshow("0",pjjimage);
                              //    cv::imshow("match",pjjimage1);
                            //      cv::imshow("best",pjjimage2);
                          //        cout<<"R21="<<R21<<endl;
                              //    cout<<"t21="<<t21<<endl;
                              //   cv::waitKey(0);
}

                          }
                    }

                    cout<<"number_point1="<<point1.size()<<endl;
                    //cout<<"number_point2="<<point2.size()<<endl;
                   // cout<<"number_point3="<<point3.size()<<endl;

                    if(point1.size()){
                    for(int i=0;i<point1.size();i=i+50){
                           cv::Point xy(point1[i].second,point1[i].first);
                          cv::Point best_uv(point2[i].second,point2[i].first);
                          cv::Point uv(point3[i].second,point3[i].first);

                          //cv::Point y(47,470);
                          std::pair<float,float> y_x=point1[i];
                          std::pair<float,float> best_v_u=point2[i];
                          std::pair<float,float> v_u=point3[i];
                   //   if(abs(bilinear<float>(kf2->GradTheta,v_u.first,v_u.second)-kf->GradTheta.at<float>(y_x.first,y_x.second))<10){
                        // cout<<"angle="<<abs(bilinear<float>(kf2->GradTheta,v_u.first,v_u.second)-kf->GradTheta.at<float>(y_x.first,y_x.second))<<endl;
                    //   cout<<"xy"<<i<<"="<<"("<<y_x.second<<","<<y_x.first<<")"<<"/photo="<<bilinear<uchar>(kf->im_,y_x.first,y_x.second)<<"gradit="<<bilinear<float>(kf->GradImg,y_x.first,y_x.second)<<endl;
                        // cout<<"(u,v)"<<i<<"="<<"("<<v_u.second<<","<<v_u.first<<")"<<"/photo="<<bilinear<uchar>(kf2->im_,v_u.first,v_u.second)<<"gradit="<<bilinear<float>(kf2->GradImg,v_u.first,v_u.second)<<endl;
                        //  cout<<"best_uv="<<"("<<best_v_u.second-v_u.second<<","<<best_v_u.first-v_u.first<<")"<<"/photo="<<bilinear<uchar>(kf2->im_,best_v_u.first,best_v_u.second)<<"gradit="<<bilinear<float>(kf2->GradImg,best_v_u.first,best_v_u.second)<<endl;
                         cv::circle(pjjimage,xy,4,cv::Scalar(255,255,255),1,8,0);
                         //   cv::circle(pjjimage,xy,4,cv::Scalar(255,255,255),1,8,0);
                          //cv::circle(pjjimage,y,4,cv::Scalar(255,255,255),1,8,0);
                          //cv::circle(pjjimage,cv::Point(0,10),10,cv::Scalar(255,255,255),1,8,0);
                          //cv::line(pjjimage,cv::Point(pjjimage.rows,0),yx,cv::Scalar(255,255,255),1,8,0);
                          //cv::line(pjjimage1,cv::Point(pjjimage1.rows,pjjimage1.cols),best_vu,cv::Scalar(255,255,255),1,8,0);
                         // float v=-((ap[i]/bp[i])*pjjimage1.cols+cp[i]/bp[i]);
                         // cout<<"(pjjv,pjju)"<<"("<<v<<","<<pjjimage1.cols/2<<")"<<endl;
                        cv::line(pjjimage1,cv::Point(0,-(cp[i]/bp[i])),cv::Point(pjjimage1.cols,-((ap[i]/bp[i])*pjjimage1.cols+cp[i]/bp[i])),cv::Scalar(255,255,255),1,8,0);
                       // cv::line(pjjimage2,cv::Point(0,-(cp[i]/bp[i])),cv::Point(pjjimage1.cols,-((ap[i]/bp[i])*pjjimage1.cols+cp[i]/bp[i])),cv::Scalar(255,255,255),1,8,0);
                     //     cv::circle(pjjimage1,uv,4,cv::Scalar(255,255,255),1,8,0);
                           cv::circle(pjjimage1,uv,4,cv::Scalar(255,255,255),1,8,0);
                          cv::circle(pjjimage2,best_uv,4,cv::Scalar(255,255,255),1,8,0);
                       //    cv::circle(pjjimage2,best_uv,4,cv::Scalar(255,255,255),1,8,0);
                           //cv::imshow("1",pjjimage);

                         //  cv::imshow("2",pjjimage1);
                            //cv::imshow("3",kf->GradImg);
                            //cv::imshow("4",kf->GradTheta);
                          // cv::imshow("0",pjjimage);
                       //    cv::imshow("match",pjjimage1);
                        //   cv::imshow("best",pjjimage2);
                        //   cout<<"R21="<<R21<<endl;
                        //   cout<<"t21="<<t21<<endl;
}
         //        }
                    cv::imshow("0",pjjimage);
                    cv::imshow("match",pjjimage1);
                    cv::imshow("best",pjjimage2);
                    //cout<<"R21="<<R21<<endl;
                    cout<<"t21="<<t21<<endl;
                   cv::waitKey(0);
                    }
      //  cv::waitKey(0);


          }
****************************************************/


  int number=0;
        for(int y = 0+2; y < image.rows-2; y+=2)
        {
          for(int x = 0+2; x< image.cols-2; x+=3)
          {
              if(kf->GradImg.ptr<float>(y)[x]<lambdaG){continue;}
        //    if(kf->GradImg.at<float>(y,x) < lambdaG){continue;}
              float pixel=(float)kf->im_.ptr<uchar>(y)[x];
              float t_pi=kf->GradTheta.ptr<float>(y)[x];
            //float pixel =(float) image.at<uchar>(y,x); //maybe it should be cv::Mat
          // cout<<"(x,y)="<<x<<","<<y<<endl;
            std::vector<depthHo> depth_ho;
           depth_ho.clear();
                 for(size_t j=0;j<closestMatches.size(); j++)
            {
                cv::Mat F12 = F[j];
               ORB_SLAM2::KeyFrame* kf2 = closestMatches[ j ];
             //   float cos=vec_cos[j];
              //  if(cos<0.998){continue;}
                //cout<<"duijiao="<<duijiao<<endl;
                float best_u(0.0),best_v(0.0),pjju(0.0),pjjv(0.0),a=0.0,b=0.0,c=0.0;
                depthHo dh;


          // EpipolarSearch(kf, kf2, x, y, pixel, min_invdepth, max_invdepth, &dh,F12,pjju,pjjv,best_u,best_v,t_pi,a,b,c);
            EpipolarSearch_after(kf, kf2, x, y, pixel, min_invdepth, max_invdepth, &dh,F12,pjju,pjjv,best_u,best_v,t_pi,a,b,c);
             //  EpipolarSearch_h(kf, kf2, x, y, pixel, min_invdepth, max_invdepth, &dh,F12,pjju,pjjv,best_u,best_v,t_pi,a,b,c);


                if (dh.supported&&dh.depth>0.01)
               {
                      depth_ho.push_back(dh);
                  }
                 }
        if(depth_ho.size()){
             // cout<<"depth_ho.size="<<depth_ho.size()<<endl;
            depthHo dh_temp;

               InverseDepthHypothesisFusion(depth_ho, dh_temp);

                if(dh_temp.supported&&dh_temp.depth>0.01)//&&dh_temp.sigma/dh_temp.depth<0.01)
                {
                // cout<<dh_temp.depth<<"dh_temp.sagma="<<dh_temp.sigma<<endl;
                  kf->depth_map_.at<float>(y,x) = dh_temp.depth;   //  used to do IntraKeyFrameDepthChecking
                  kf->depth_sigma_.at<float>(y,x) = dh_temp.sigma;
                  number++;
                }

          }
        }
        }



     std::cout<<"IntraKeyFrameDepthChecking"<<"num_point_cnt"<<number<<std::endl;
     IntraKeyFrameDepthChecking( kf->depth_map_,  kf->depth_sigma_, kf->GradImg);
    // saveMatToCsv(kf->depth_map_,"depth_pjj1.csv");
     kf->semidense_begin_inter=true;
     //cv::waitKey(0);
#ifndef InterKeyFrameChecking
    // octomap::OcTree tree(0.01);
     for(int y = 0+2; y < image.rows-2; y++)
     {
      //   float *kfdepth_map=&kf->depth_map_.data[y*kf->depth_map_.step];
       for(int x = 0+2; x< image.cols-2; x++)
       {

         if(kf->depth_map_.ptr<float>(y)[x]< 0.01) continue;

         float inv_d = kf->depth_map_.ptr<float>(y)[x];
         float Z = 1/inv_d ;
         float X = Z *(x- kf->cx ) / kf->fx;
         float Y = Z*(y- kf->cy ) / kf->fy;

         cv::Mat Pc = (cv::Mat_<float>(4,1) << X, Y , Z, 1); // point in camera frame.
         cv::Mat Twc = kf->GetPoseInverse();
         cv::Mat pos = Twc * Pc;

         kf->SemiDensePointSets_.at<float>(y,3*x+0) = pos.at<float>(0);
         kf->SemiDensePointSets_.at<float>(y,3*x+1) = pos.at<float>(1);
         kf->SemiDensePointSets_.at<float>(y,3*x+2) = pos.at<float>(2);

       }
     }

     kf->semidense_flag_ = true;    // set this flag after inter-KeyFrame checked

#endif
/********dian yun xian shi***************
    // pcl::visualization::CloudViewer viewer("cloud viewer");
     pcl::PointCloud<pcl::PointXYZRGB> cloudPtr;
    cloudPtr.width=640;
    cloudPtr.height=480;
    cloudPtr.points.resize(cloudPtr.width*cloudPtr.height);
    float i_=0;
for(size_t y=2;y<image.rows-2;++y){
 for(size_t x=2;x<image.cols-2;++x){
     cloudPtr.points[i_].x=kf->SemiDensePointSets_.at<float>(y,3*x+0) ;
     cloudPtr.points[i_].y=kf->SemiDensePointSets_.at<float>(y,3*x+1);
     cloudPtr.points[i_].z=kf->SemiDensePointSets_.at<float>(y,3*x+2) ;
     cloudPtr.points[i_].r=kf->rgb_.at<uchar>(y,3*x+2) / 255.0;
     cloudPtr.points[i_].g= kf->rgb_.at<uchar>(y,3*x+1) / 255.0;
     cloudPtr.points[i_].b= kf->rgb_.at<uchar>(y,3*x) / 255.0;
     i_++;
 }
}
pcl::io::savePCDFileASCII("test_pcd.pcd",cloudPtr);
 pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
 pcl::io::loadPCDFile("test_pcd.pcd",*cloud);
 pcl::visualization::CloudViewer viewer1("pjj");
 viewer1.showCloud(cloud);
********************************/



  }

#ifdef InterKeyFrameChecking
   for(size_t i =0;i < vpKFs.size();++i)
  {
      //std::vector<ORB_SLAM2::KeyFrame*> vpKFs_pjj;
      ORB_SLAM2::KeyFrame* kf = vpKFs[i];
      if(kf->isBad() || kf->semidense_flag_)continue;
      if( kf->semidense_begin_inter==false)continue;
      int biaozhi=0;
      InterKeyFrameDepthChecking(kf,biaozhi);
if(biaozhi==1){
   IntraKeyFrameDepthChecking1( kf->depth_map_,  kf->depth_sigma_, kf->GradImg);
      for(int y = 0+2; y < kf->im_.rows-2; y++)
      {
         //  uchar *kfdepth1=&kf->depth_map_.data[y*kf->depth_map_.step];
        for(int x = 0+2; x< kf->im_.cols-2; x++)
        {

       //   if(kf->depth_map_.ptr<float>(y)[x]< 0.01) continue;
  if(kf->depth_map_.ptr<float>(y)[x]<0.01)continue;
       //   float inv_d =kf->depth_map_.ptr<float>(y)[x];
  float inv_d=kf->depth_map_.ptr<float>(y)[x];
          float Z = 1/inv_d ;
          float X = Z *(x- kf->cx ) / kf->fx;
          float Y = Z*(y- kf->cy ) / kf->fy;

          cv::Mat Pc = (cv::Mat_<float>(4,1) << X, Y , Z, 1); // point in camera frame.
          cv::Mat Twc = kf->GetPoseInverse();
          cv::Mat pos = Twc * Pc;

          kf->SemiDensePointSets_.at<float>(y,3*x+0) = pos.at<float>(0);
          kf->SemiDensePointSets_.at<float>(y,3*x+1) = pos.at<float>(1);
          kf->SemiDensePointSets_.at<float>(y,3*x+2) = pos.at<float>(2);
        }
      }
      kf->semidense_flag_ = true;
     //IntraKeyFrameDepthChecking1( kf->depth_map_,  kf->depth_sigma_, kf->GradImg);
}
  }
#endif
}

void ProbabilityMapping::StereoSearchConstraints(ORB_SLAM2::KeyFrame* kf, float* min_invdepth, float* max_invdepth){
  std::vector<float> orb_depths = kf->GetAllPointDepths(20);
for(auto &i:orb_depths){i=1/i;}
  float sum = std::accumulate(orb_depths.begin(), orb_depths.end(), 0.0);
  float mean = sum / orb_depths.size();

  std::vector<float> diff(orb_depths.size());
  std::transform(orb_depths.begin(), orb_depths.end(), diff.begin(), std::bind2nd(std::minus<float>(), mean));
  float variance = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0)/orb_depths.size();
  float stdev = std::sqrt(variance);

float a=mean-2*stdev;//
 if(a<0){ cout<<"mean="<<mean<<"stdev="<<stdev<<endl;}
if(a<0){a=0.01;}
  *max_invdepth = mean + 2 * stdev;
  *min_invdepth = a;//mean - 2 * stdev;

}
/*

 void ProbabilityMapping::EpipolarSearch(ORB_SLAM2::KeyFrame* kf1, ORB_SLAM2::KeyFrame *kf2, const int x, const int y, float pixel,
    float min_invdepth, float max_invdepth, depthHo *dh,cv::Mat F12,float& pjju,float& pjjv,float& best_u,float& best_v,float th_pi,float&  a,float& b,float& c)
{
 //if( kf1->GradImg.at<float>(y,x) < lambdaG){return;}
 a = x*F12.at<float>(0,0)+y*F12.at<float>(1,0)+F12.at<float>(2,0);
  b = x*F12.at<float>(0,1)+y*F12.at<float>(1,1)+F12.at<float>(2,1);
  c = x*F12.at<float>(0,2)+y*F12.at<float>(1,2)+F12.at<float>(2,2);

  cv::Mat R1w = kf1->GetRotation();
  cv::Mat t1w = kf1->GetTranslation();
  cv::Mat R2w = kf2->GetRotation();
  cv::Mat t2w = kf2->GetTranslation();

  cv::Mat R12 = R1w*R2w.t();
  cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
  cv::Mat R21=R2w*R1w.t();
  cv::Mat t21=-R2w*R1w.t()*t1w+t2w;

  if((a/b)< -4 || a/b> 4) return;   // if epipolar direction is approximate to perpendicular, we discard it.  May be product wrong match.
  float old_err = 3000.0,second_err=30000;
  float best_photometric_err = 0.0;
  float best_gradient_modulo_err = 0.0;

  int best_pixel = 0,second_pixel=0;


  int uj_plus,uj_minus;
  float vj,vj_plus,vj_minus;
  float g, q,denomiator ,ustar , ustar_var;
    float umin(0.0),umax(0.0);
   if((max_invdepth-min_invdepth)<0){cout<<"error_max_invdepth-min_invdepth"<<endl;}
  GetSearchRange(umin,umax,x,y,min_invdepth,max_invdepth,kf1,kf2);
  if(umax==umin){return;}


  float t1=umin,t2=umax;
  int detau=30;
  float d=(min_invdepth+max_invdepth)/2;
   fanwei_u_homography(x,y,d,R21, t21,umin,umax, detau,kf1);
if(t1<umin&&umin-t1>640){cout<<umin<<"  "<<t1<<endl;}


 // cout<<"t1,t2="<<t2<<","<<t1<<"umax,umin="<<umax<<","<<umin<<endl;
  int number_umin_umax=0;
  //vector<int > vec_u;
  //for(int uj = 0; uj < image.cols; uj++)// FIXME should use  min and max depth
  for(int uj = std::floor(umin); uj < std::ceil(umax)+1; uj++)// FIXME should use  min and max depth
  {

     vj =-( (a/b)*uj+(c/b));
    if(vj<0|| vj > kf2->im_.rows ){continue;}
    if(uj<0||uj>kf2->im_.cols){continue;}


    // condition 1:

 //   if( kf2->GradImg.at<float>(round(vj),uj) < lambdaG){continue;}
   if(bilinear<float>(kf2->GradImg,vj,uj)<lambdaG){continue;}
    // condition 2:
    float th_epipolar_line = cv::fastAtan2(-a/b,1);
   // float th_epipolar_line1=cv::fastAtan2(uj,vj);
    //cout<<"th_ep"<<th_epipolar_line<<"th_ep1="<<th_epipolar_line1<<endl;
 //   float temp_gradth =  kf2->GradTheta.at<float>(vj,uj) ;
    float temp_gradth=bilinear<float>(kf2->GradTheta,vj,uj);
 //float aqwe=1.6,bqwe=1.5,cqwe=1.4,dqwe=1.2,eqwe=1.7;
//  cout<<round(aqwe)<<"  "<<round(bqwe)<<"  "<<round(cqwe)<<"  "<<round(dqwe)<<"  "<<round(eqwe)<<endl;
//    cout<<"temp_gradth="<<kf2->GradTheta.at<float>((int)vj,uj)<<temp_gradth<<endl;
    if( temp_gradth > 270) temp_gradth =  temp_gradth - 360;
    if( temp_gradth > 90 &&  temp_gradth<=270)
      temp_gradth =  temp_gradth - 180;
   if(th_epipolar_line>270) th_epipolar_line = th_epipolar_line - 360;
    if(th_epipolar_line>90&&th_epipolar_line<=270){
        th_epipolar_line=th_epipolar_line-180;
    }
   if(abs(abs(temp_gradth - th_epipolar_line) - 90)< 10 ){  continue;}


    // condition 3:

 //  if(abs( kf2->GradTheta.at<float>(vj,uj) -  th_pi ) > lambdaTheta)continue;
  if(abs(bilinear<float>(kf2->GradTheta,vj,uj)-th_pi)>lambdaTheta)continue;

     float photometric_err = pixel - bilinear<uchar>(kf2->im_,-((a/b)*uj+(c/b)),uj);
    float gradient_modulo_err = kf1->GradImg.at<float>(y,x)  - bilinear<float>( kf2->GradImg,-((a/b)*uj+(c/b)),uj);

 float err = (photometric_err*photometric_err+ (gradient_modulo_err*gradient_modulo_err)/THETA);

    if(err < old_err)
    {

        second_err=old_err;
       best_pixel = uj;
      //ncc_score=NCC;
      old_err = err;
    // best_photometric_err = photometricerr;
     best_photometric_err = photometric_err;
      //best_gradient_modulo_err = gradient_moduloerr;
     best_gradient_modulo_err = gradient_modulo_err;
     // cout<<"old_err="<<old_err<<endl;
    }

    //qwe
  }

 if(old_err <500)
  {
 // float uj1=best_pixel;
    //  cout<<"old_err="<<old_err<<endl;
 // cout<<"best_photo_err2="<<best_photometric_err*best_photometric_err<<"/best_gradient_err2="<<best_gradient_modulo_err*best_gradient_modulo_err*(1/0.23)<<"/bili="<<(best_photometric_err*best_photometric_err)/old_err<<endl;
  uj_plus = best_pixel + 1;
     vj_plus = -((a/b)*uj_plus + (c/b));
    uj_minus = best_pixel - 1;
   vj_minus = -((a/b)*uj_minus + (c/b));

   if(vj_plus<0||vj_minus<0||vj_plus>kf2->im_.rows||uj_minus>kf2->im_.cols){return;}
    pjju=best_pixel;
    pjjv=-((a/b)*pjju+(c/b));


 cv::Mat grad2mag=kf2->GradImg;
 cv::Mat image=kf2->im_;
 g = (bilinear<uchar>(image,-((a/b)*uj_plus+(c/b)),uj_plus) - bilinear<uchar>(image,-((a/b)*uj_minus+(c/b)),uj_minus))/2.0;
 q= (bilinear<float>(grad2mag,-((a/b)*uj_plus+(c/b)),uj_plus) - bilinear<float>(grad2mag,-((a/b)*uj_minus+(c/b)),uj_minus))/2.0;

  denomiator = (g*g + (1/THETA)*q*q);
  ustar = best_pixel +(g*best_photometric_err + (1/THETA)*q*best_gradient_modulo_err)/denomiator;
 if(pow(ustar-best_pixel,2)>1)
{ustar=best_pixel;
}
ustar_var=1;

     best_u = ustar;
     best_v =  -( (a/b)*best_u + (c/b) );

     //ComputeInvDepthHypothesis(kf1, kf2,umin,umax, ustar, ustar_var, a, b, c, dh,x,y);
      ComputeInvDepthHypothesis(kf1, kf2,umin,umax, ustar, ustar_var, a, b, c, dh,x,y);
    //ComputeInvDepthHypothesis(kf1, kf2, best_pixel, ustar_var, a, b, c, dh,x,y);
  }


 }


*/
 void ProbabilityMapping::EpipolarSearch_after(ORB_SLAM2::KeyFrame* kf1, ORB_SLAM2::KeyFrame *kf2, const int x, const int y, float pixel,
    float min_invdepth, float max_invdepth, depthHo *dh,cv::Mat F12,float&pjju,float& pjjv,float& best_u,float& best_v,float th_pi,float&a,float&b,float&c)
{

   a = x*F12.at<float>(0,0)+y*F12.at<float>(1,0)+F12.at<float>(2,0);
   b = x*F12.at<float>(0,1)+y*F12.at<float>(1,1)+F12.at<float>(2,1);
   c = x*F12.at<float>(0,2)+y*F12.at<float>(1,2)+F12.at<float>(2,2);

  if((a/b)< -4 || a/b> 4) return;   // if epipolar direction is approximate to perpendicular, we discard it.  May be product wrong match.

  //qwe

  cv::Mat R1w = kf1->GetRotation();
  cv::Mat t1w = kf1->GetTranslation();
  cv::Mat R2w = kf2->GetRotation();
  cv::Mat t2w = kf2->GetTranslation();

  //cv::Mat R12 = R1w*R2w.t();
 // cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
  cv::Mat R21=R2w*R1w.t();
  cv::Mat t21=-R2w*R1w.t()*t1w+t2w;
  cv::Mat K2=kf2->GetCalibrationMatrix();
  cv::Mat K1 = kf1->GetCalibrationMatrix();
  cv::Mat xy1=(cv::Mat_<float>(3,1) << x, y,1.0);
  cv::Mat k2t21=K2*t21;
  //cv::Mat Rxy1=K2*R2w*R1w.t()*K1.inv()*xy1;
  //cv::Mat k2R21K1_=K2*R21*K1.inv();
  cv::Mat Rxy1=K2*R21*K1.inv()*xy1;


cv::Mat normal1=(cv::Mat_<float>(1,3)<<0,0,1);
cv::Mat H1=K2*R21*K1.inv();
cv::Mat H2=K2*t21*normal1*K1.inv();


 std::vector<cv::Mat>n_point;
 n_point.clear();
  std::vector<float> photo_kf1,photo_kf1_,gradient_kf1,vector_ncc_score;
  photo_kf1.clear();



  int halfpath_size=2;


  if(x<halfpath_size||x+halfpath_size>kf1->im_.cols||y-halfpath_size<0||y+halfpath_size>kf1->im_.rows){return;}

std::vector<float> w_pq;
float num=0;
vector<int> vec_number;
vector<pair<float,float>> vec_xy;
float sum_photo_kf1=0.0,sum_gradient_kf1=0.0;
  for(int  j=-halfpath_size;j<halfpath_size+1;++j){
  for(int i=-halfpath_size;i<halfpath_size+1;++i){

      pair<float,float> xy=make_pair(i+x,j+y);
     vec_xy.push_back(xy);
     cv::Mat xppjj=(cv::Mat_<float>(3,1)<<i+x,j+y,1);

      n_point.push_back(xppjj);

        photo_kf1.push_back(bilinear1(kf1->im_,j+y,i+x));
        sum_photo_kf1+=bilinear1(kf1->im_,j+y,i+x);

  }
  }


      sum_photo_kf1/=photo_kf1.size();



  float old_err =10000,second_err;
  float best_photometric_err = 0.0;
  float best_gradient_modulo_err = 0.0;
  float NSSD=1000000;
  float ncc_score=0.0,second_score=0.0;
  int best_census_filter=1000;
  int best_pixel = 0,second_pixel;
  std::vector<float> better_err_uj;

float gradient_err,photo_err;
int number=0;
  int  uj_plus,uj_minus;
  float vj_plus,vj_minus,vj;
  float g, q,denomiator ,ustar , ustar_var;

  float umin(0.0),umax(0.0);

  if(kf1->depth_map_first.ptr<float>(y)[x]){
     float  min_invdepth1=kf1->depth_map_first.ptr<float>(y)[x]-kf1->depth_sigma_first.ptr<float>(y)[x];
     float  max_invdepth1=kf1->depth_map_first.ptr<float>(y)[x]+kf1->depth_sigma_first.ptr<float>(y)[x];
      if(min_invdepth1>min_invdepth&&min_invdepth1<max_invdepth){min_invdepth=min_invdepth1;}
      if(max_invdepth1<max_invdepth&&max_invdepth1>min_invdepth){max_invdepth=max_invdepth1;}
  }

  GetSearchRange(umin,umax,x,y,min_invdepth,max_invdepth,kf1,kf2);
   if(umin==umax){return;}

   for(int uj = std::floor(umin); uj < std::ceil(umax)+1; ++uj){
            vj =-( (a/b)*uj+(c/b));

          if(uj<halfpath_size||uj+halfpath_size>kf1->im_.cols||vj-halfpath_size<0||vj+halfpath_size>kf1->im_.rows){continue;}
            if(vj<0|| vj > kf2->im_.rows ){continue;}


            // condition 1:
            //if( kf2->GradImg.at<float>(vj,uj) < lambdaG){continue;}
          if(bilinear<float>(kf2->GradImg,vj,uj)<lambdaG){continue;}
            // condition 2:
           float th_epipolar_line = cv::fastAtan2(-a/b,1);
            float temp_gradth =  bilinear<float>(kf2->GradTheta,vj,uj) ;
            if( temp_gradth > 270) temp_gradth =  temp_gradth - 360;
            if( temp_gradth > 90 &&  temp_gradth<=270)
              temp_gradth =  temp_gradth - 180;
           if(th_epipolar_line>270) th_epipolar_line = th_epipolar_line - 360;
            if(th_epipolar_line>90&&th_epipolar_line<=270){th_epipolar_line=th_epipolar_line-180;}
            if(abs(abs(temp_gradth - th_epipolar_line) - 90)< 10 ){  continue;}
               // condition 3:
              if(abs(bilinear<float>( kf2->GradTheta,vj,uj) -  th_pi ) > lambdaTheta)continue;

            //if(abs(th_grad - ( th_pi + th_rot )) > lambdaTheta)continue;
           //   float number_depth=round((uj-floor(umin))/3);
         //     if(number_depth>number_h-1)number_depth=number_h-1;
           float s_depth=(k2t21.at<float>(0,0)-k2t21.at<float>(2,0)*uj)/(Rxy1.at<float>(2,0)*uj-Rxy1.at<float>(0,0)+1e-10);

            std::vector<float> photo_kf2,photo_kf2_,gradient_kf2;
           vector<int> vec_number1;
               photo_kf2.clear();
             if(n_point.size()==0){continue;}
               std::vector<float>w1_pq;
                //cv::Mat H=H1+H2_up/(s_depth*h2_down);
               cv::Mat H=H1+H2/s_depth;
              //  cv::Mat pjj_=K2*R2w-H*K1*R1w;
               // cv::Mat pjj_t=K2*t2w-H*K1*t1w;
              //  cout<<"pjj_="<<pjj_<<"pjj_t="<<pjj_t<<endl;
                vector<pair<float,float>> vec_uv;
                float sum_photo_kf2=0.0,sum_gradient_kf2=0.0,num_ramuda=0;
               for(size_t i=0;i<n_point.size();++i){

                cv::Mat uvpjj=H*n_point[i];
             float u_pjj1=uvpjj.at<float>(0,0)/uvpjj.at<float>(2,0);
             float v_pjj1=uvpjj.at<float>(1,0)/uvpjj.at<float>(2,0);

             photo_kf2.push_back(bilinear1(kf2->im_,v_pjj1,u_pjj1));
             sum_photo_kf2+=bilinear1(kf2->im_,v_pjj1,u_pjj1);

           }

               sum_photo_kf2/=photo_kf2.size();
             float pjjfenzhi=0.0,pjjfenmu1=0.0,pjjfenmu2=0.0,gradientfenzhi=0.0,gradientfenmu1=0.0,gradientfenmu2=0.0;
            int census_filter=0;
              float nssd_score=0;
             float err=0.0,err2=0;
             float err_support_weight=0,err_support_weight1=0;
            // cout<<"number="<<photo_kf1.size()<<endl;
            // if(vec_uv.size()!=vec_xy.size()){cout<<"error photo_kf1 photo_kf2 is not same"<<endl;}

          for(size_t i=0;i<photo_kf1.size();i++){
          pjjfenzhi+=(photo_kf1[i]-sum_photo_kf1)*(photo_kf2[i]-sum_photo_kf2);
          pjjfenmu1+=pow((photo_kf1[i]-sum_photo_kf1),2);
             //pjjfenmu2+=pow(w1_pq[i]*(photo_kf2[i]-sum_photo_kf2),2);
            pjjfenmu2+=pow((photo_kf2[i]-sum_photo_kf2),2);


           }


          float NCC=pjjfenzhi/(std::sqrt(pjjfenmu1*pjjfenmu2)+1e-10);//+gradientfenzhi/(std::sqrt(gradientfenmu1*gradientfenmu2)*THETA+1e-10);
           // float NCC=gradientfenzhi/(std::sqrt(gradientfenmu1*gradientfenmu2)+1e-10);
           // cout<<"NCC="<<NCC<<endl;
       if(NCC>ncc_score){
           second_pixel=best_pixel;
          second_score=ncc_score;
        best_pixel = uj;
         ncc_score=NCC;
  }

        }//uj umin~umax


 if((ncc_score-second_score)/(ncc_score+1e-10)<0.01){return;}

if(ncc_score>0.85){
    pjju=best_pixel;
    pjjv=-((a/b)*pjju+(c/b));
    best_u = pjju;
    best_v =  -( (a/b)*best_u + (c/b) );
    float photometric_err = pixel - bilinear<uchar>(kf2->im_,best_v, best_u);
    float gradient_modulo_err = kf1->GradImg.at<float>(y,x)  - bilinear<float>( kf2->GradImg,best_v, best_u);
    float err = ((photometric_err*photometric_err  + (gradient_modulo_err*gradient_modulo_err)/THETA)*(1/(sigmaI*sigmaI)));
    ofstream file;
    file.open("err.txt",ios::app);
    file<<err;

       ComputeInvDepthHypothesis(kf1, kf2,umin,umax, best_pixel, 1, a, b, c, dh,x,y);
}

}
 /*
 void ProbabilityMapping::EpipolarSearch_h(ORB_SLAM2::KeyFrame* kf1, ORB_SLAM2::KeyFrame *kf2, const int x, const int y, float pixel,
    float min_invdepth, float max_invdepth, depthHo *dh,cv::Mat F12,float&pjju,float& pjjv,float& best_u,float& best_v,float th_pi,float&a,float&b,float&c)
{

   a = x*F12.at<float>(0,0)+y*F12.at<float>(1,0)+F12.at<float>(2,0);
   b = x*F12.at<float>(0,1)+y*F12.at<float>(1,1)+F12.at<float>(2,1);
   c = x*F12.at<float>(0,2)+y*F12.at<float>(1,2)+F12.at<float>(2,2);

  if((a/b)< -4 || a/b> 4) return;   // if epipolar direction is approximate to perpendicular, we discard it.  May be product wrong match.

  //qwe

  cv::Mat R1w = kf1->GetRotation();
  cv::Mat t1w = kf1->GetTranslation();
  cv::Mat R2w = kf2->GetRotation();
  cv::Mat t2w = kf2->GetTranslation();

  cv::Mat R12 = R1w*R2w.t();
  cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
  cv::Mat R21=R2w*R1w.t();
  cv::Mat t21=-R2w*R1w.t()*t1w+t2w;

  cv::Mat K2=kf2->GetCalibrationMatrix();
  cv::Mat K1 = kf1->GetCalibrationMatrix();
  cv::Mat xy1=(cv::Mat_<float>(3,1) << x, y,1.0);
  cv::Mat k2t21=K2*(-R2w*R1w.t()*t1w+t2w);
  cv::Mat Rxy1=K2*R2w*R1w.t()*K1.inv()*xy1;


  cv::Mat normal1=(cv::Mat_<float>(3,1)<<0,0,1);
  cv::Mat a_pjj=R2w*R1w.t();
    cv::Mat b_pjj=kaifang(a_pjj);
    cv::Mat normal2=b_pjj.t()*normal1;
    cv::Mat normal=normal2.t();

    cv::Mat H1=K2*R2w*R1w.t()*K1.inv();
    cv::Mat H2_up=-K2*R2w*R1w.t()*t1w*normal*K1.inv()+K2*t2w*normal*K1.inv();
    cv::Mat H2_down=normal*K1.inv()*xy1;
    float h2_down=H2_down.at<float>(0,0);


 std::vector<cv::Mat>n_point;
 n_point.clear();
  std::vector<float> photo_kf1,photo_kf1_,gradient_kf1,vector_ncc_score;
  photo_kf1.clear();



  int halfpath_size=2;


  if(x<halfpath_size||x+halfpath_size>kf1->im_.cols||y-halfpath_size<0||y+halfpath_size>kf1->im_.rows){return;}
//cv::Mat xy=(cv::Mat_<float>(2,1)<<x,y);
std::vector<float> w_pq;
float num=0;
vector<int> vec_number;
vector<pair<float,float>> vec_xy;
float sum_photo_kf1=0.0,sum_gradient_kf1=0.0;
  for(int  j=-halfpath_size;j<halfpath_size+1;++j){
  for(int i=-halfpath_size;i<halfpath_size+1;++i){

      pair<float,float> xy=make_pair(i+x,j+y);
     vec_xy.push_back(xy);
            cv::Mat xppjj=(cv::Mat_<float>(3,1)<<i+x,j+y,1);

      n_point.push_back(xppjj);

        photo_kf1.push_back(bilinear1(kf1->im_,j+y,i+x));
        sum_photo_kf1+=bilinear1(kf1->im_,j+y,i+x);
  ;
  }
  }




      sum_photo_kf1/=photo_kf1.size();

//qwer
  float old_err =10000,second_err;
  float best_photometric_err = 0.0;
  float best_gradient_modulo_err = 0.0;
  float NSSD=1000000;
  float ncc_score=0.0,second_score=0.0;
  int best_census_filter=1000;
  int best_pixel = 0,second_pixel;
  std::vector<float> better_err_uj;

float gradient_err,photo_err;
int number=0;
  int  uj_plus,uj_minus;
  float vj_plus,vj_minus,vj;
  float g, q,denomiator ,ustar , ustar_var;

  float umin(0.0),umax(0.0);

  if(kf1->depth_map_first.ptr<float>(y)[x]){
     float  min_invdepth1=kf1->depth_map_first.ptr<float>(y)[x]-kf1->depth_sigma_first.ptr<float>(y)[x];
     float  max_invdepth1=kf1->depth_map_first.ptr<float>(y)[x]+kf1->depth_sigma_first.ptr<float>(y)[x];
      if(min_invdepth1>min_invdepth&&min_invdepth1<max_invdepth){min_invdepth=min_invdepth1;}
      if(max_invdepth1<max_invdepth&&max_invdepth1>min_invdepth){max_invdepth=max_invdepth1;}
  }

  GetSearchRange(umin,umax,x,y,min_invdepth,max_invdepth,kf1,kf2);
   if(umin==umax){return;}


   int detau=20;
   float d=(min_invdepth+max_invdepth)/2;
  // cout<<"H-R21="<<H-R21<<endl;
    fanwei_u_homography(x,y,d,R21, t21,umin,umax, detau,kf1);


   for(int uj = std::floor(umin); uj < std::ceil(umax)+1; ++uj){
            vj =-( (a/b)*uj+(c/b));

          if(uj<halfpath_size||uj+halfpath_size>kf1->im_.cols||vj-halfpath_size<0||vj+halfpath_size>kf1->im_.rows){continue;}
            if(vj<0|| vj > kf2->im_.rows ){continue;}


            // condition 1:
            //if( kf2->GradImg.at<float>(vj,uj) < lambdaG){continue;}
          if(bilinear<float>(kf2->GradImg,vj,uj)<lambdaG){continue;}
            // condition 2:
           float th_epipolar_line = cv::fastAtan2(-a/b,1);
            float temp_gradth =  bilinear<float>(kf2->GradTheta,vj,uj) ;
            if( temp_gradth > 270) temp_gradth =  temp_gradth - 360;
            if( temp_gradth > 90 &&  temp_gradth<=270)
              temp_gradth =  temp_gradth - 180;
           if(th_epipolar_line>270) th_epipolar_line = th_epipolar_line - 360;
            if(th_epipolar_line>90&&th_epipolar_line<=270){th_epipolar_line=th_epipolar_line-180;}
            if(abs(abs(temp_gradth - th_epipolar_line) - 90)< 10 ){  continue;}
               // condition 3:
              if(abs(bilinear<float>( kf2->GradTheta,vj,uj) -  th_pi ) > lambdaTheta)continue;

            //if(abs(th_grad - ( th_pi + th_rot )) > lambdaTheta)continue;
           //   float number_depth=round((uj-floor(umin))/3);
         //     if(number_depth>number_h-1)number_depth=number_h-1;
           float s_depth=(k2t21.at<float>(0)-k2t21.at<float>(2)*uj)/(Rxy1.at<float>(2)*uj-Rxy1.at<float>(0)+1e-10);


            std::vector<float> photo_kf2,photo_kf2_,gradient_kf2;
           vector<int> vec_number1;
               photo_kf2.clear();
             if(n_point.size()==0){continue;}
               std::vector<float>w1_pq;
                cv::Mat H=H1+H2_up/(s_depth*h2_down);
              //  cv::Mat pjj_=K2*R2w-H*K1*R1w;
               // cv::Mat pjj_t=K2*t2w-H*K1*t1w;
              //  cout<<"pjj_="<<pjj_<<"pjj_t="<<pjj_t<<endl;
                vector<pair<float,float>> vec_uv;
                float sum_photo_kf2=0.0,sum_gradient_kf2=0.0,num_ramuda=0;
               for(size_t i=0;i<n_point.size();++i){

                cv::Mat uvpjj=H*n_point[i];
             float u_pjj1=uvpjj.at<float>(0,0)/uvpjj.at<float>(2,0);
             float v_pjj1=uvpjj.at<float>(1,0)/uvpjj.at<float>(2,0);

             photo_kf2.push_back(bilinear1(kf2->im_,v_pjj1,u_pjj1));
             sum_photo_kf2+=bilinear1(kf2->im_,v_pjj1,u_pjj1);

           }

               sum_photo_kf2/=photo_kf2.size();

             float pjjfenzhi=0.0,pjjfenmu1=0.0,pjjfenmu2=0.0,gradientfenzhi=0.0,gradientfenmu1=0.0,gradientfenmu2=0.0;
            int census_filter=0;
              float nssd_score=0;
             float err=0.0,err2=0;
             float err_support_weight=0,err_support_weight1=0;
            // cout<<"number="<<photo_kf1.size()<<endl;
            // if(vec_uv.size()!=vec_xy.size()){cout<<"error photo_kf1 photo_kf2 is not same"<<endl;}

          for(size_t i=0;i<photo_kf1.size();i++){

               pjjfenzhi+=(photo_kf1[i]-sum_photo_kf1)*(photo_kf2[i]-sum_photo_kf2);

          pjjfenmu1+=pow((photo_kf1[i]-sum_photo_kf1),2);

            pjjfenmu2+=pow((photo_kf2[i]-sum_photo_kf2),2);
           }
          float NCC=pjjfenzhi/(std::sqrt(pjjfenmu1*pjjfenmu2)+1e-10);

       if(NCC>ncc_score){
           second_pixel=best_pixel;
          second_score=ncc_score;
        best_pixel = uj;
         ncc_score=NCC;
  }

        }//uj umin~umax


 if((ncc_score-second_score)/(ncc_score+1e-10)<0.01){return;}

if(ncc_score>0.85){
    pjju=best_pixel;
    pjjv=-((a/b)*pjju+(c/b));
    best_u = pjju;
    best_v =  -( (a/b)*best_u + (c/b) );
       ComputeInvDepthHypothesis(kf1, kf2,umin,umax, best_pixel, 1, a, b, c, dh,x,y);
}
   //  if(number!=0){cout<<number<<endl;}
}

 */

 void ProbabilityMapping::EpipolarSearch_after1(ORB_SLAM2::KeyFrame* kf1, ORB_SLAM2::KeyFrame *kf2, const int x, const int y, float pixel,
    float min_invdepth, float max_invdepth, depthHo *dh,cv::Mat F12,float&pjju,float& pjjv,float& best_u,float& best_v,float th_pi,float&a,float&b,float&c)
{

   a = x*F12.at<float>(0,0)+y*F12.at<float>(1,0)+F12.at<float>(2,0);
   b = x*F12.at<float>(0,1)+y*F12.at<float>(1,1)+F12.at<float>(2,1);
   c = x*F12.at<float>(0,2)+y*F12.at<float>(1,2)+F12.at<float>(2,2);

  if((a/b)< -4 || a/b> 4) return;   // if epipolar direction is approximate to perpendicular, we discard it.  May be product wrong match.

  //qwe

  cv::Mat R1w = kf1->GetRotation();
  cv::Mat t1w = kf1->GetTranslation();
  cv::Mat R2w = kf2->GetRotation();
  cv::Mat t2w = kf2->GetTranslation();

  //cv::Mat R12 = R1w*R2w.t();
 // cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;
  cv::Mat R21=R2w*R1w.t();
  cv::Mat t21=-R2w*R1w.t()*t1w+t2w;
  cv::Mat K2=kf2->GetCalibrationMatrix();
  cv::Mat K1 = kf1->GetCalibrationMatrix();
  cv::Mat xy1=(cv::Mat_<float>(3,1) << x, y,1.0);
  cv::Mat k2R21K1_=K2*R21*K1.inv();
  cv::Mat k2t21=K2*t21;
  cv::Mat Rxy1=k2R21K1_*xy1;

  cv::Mat a_pjj=R2w*R1w.t();
    cv::Mat b_pjj=kaifang(a_pjj);
  cv::Mat R3w=R1w*b_pjj;
  cv::Mat  t3w=R3w*R1w.t()*t1w+1/2*R3w.t()*R1w*R21*t21;
cv::Mat R31=R3w*R1w.t();
cv::Mat t31=-R3w*R1w.t()*t1w+t3w;
cv::Mat kR31K_= K1*R31*K1.inv();
cv::Mat kR31K_xy1=kR31K_*xy1;
cv::Mat kt31=K1*t31;
cv::Mat kR32K_=K1*R3w*R2w.t()*K1.inv();
cv::Mat kt32=K1*(-R3w*R2w.t()*t2w+t3w);


  //qwr

  //qwe

  int halfpath_size=2;


  if(x<halfpath_size||x+halfpath_size>kf1->im_.cols||y-halfpath_size<0||y+halfpath_size>kf1->im_.rows){return;}
//qwer
  float old_err = 10000000.0,second_err;
  float best_photometric_err = 0.0;
  float best_gradient_modulo_err = 0.0;
  float NSSD=1000000;
  float ncc_score=0.0,second_score=0.0;
  int best_census_filter=1000;
  int best_pixel = 0,second_pixel;
  std::vector<float> better_err_uj;



  int vj,uj_plus,vj_plus,uj_minus,vj_minus;
  float g, q,denomiator ,ustar , ustar_var;

  float umin(0.0),umax(0.0);
  if(kf1->depth_map_first.at<float>(y,x)){
    min_invdepth=kf1->depth_map_first.at<float>(y,x)-kf1->depth_sigma_first.at<float>(y,x);
    max_invdepth=kf1->depth_map_first.at<float>(y,x)+kf1->depth_sigma_first.at<float>(y,x);
  }

  GetSearchRange(umin,umax,x,y,min_invdepth,max_invdepth,kf1,kf2);
  //for(int uj = 0; uj < image.cols; uj++)// FIXME should use  min and max depth

   for(int uj = std::floor(umin); uj < std::ceil(umax)+1; ++uj){
            vj =-(int)( (a/b)*uj+(c/b));

          if(uj<halfpath_size||uj+halfpath_size>kf1->im_.cols||vj-halfpath_size<0||vj+halfpath_size>kf1->im_.rows){continue;}
            if(vj<0|| vj > kf2->im_.rows ){continue;}

            // condition 1:
            uchar* kf2gradimg_row_ptr=kf2->GradImg.ptr(vj);
            //if( kf2->GradImg.at<float>(vj,uj) < lambdaG){continue;}
            if(kf2gradimg_row_ptr[uj]<lambdaG){continue;}

            // condition 2:
            float th_epipolar_line = cv::fastAtan2(-a/b,1);
            float temp_gradth =  kf2->GradTheta.at<float>(vj,uj) ;
            if( temp_gradth > 270) temp_gradth =  temp_gradth - 360;
            if( temp_gradth > 90 &&  temp_gradth<=270)
              temp_gradth =  temp_gradth - 180;
            if(th_epipolar_line>270) th_epipolar_line = th_epipolar_line - 360;
            if(abs(abs(temp_gradth - th_epipolar_line) - 90)< 10 ){  continue;}
              if(abs( kf2->GradTheta.at<float>(vj,uj) -  th_pi ) > lambdaTheta)continue;
            // condition 3:
            //if(abs(th_grad - ( th_pi + th_rot )) > lambdaTheta)continue;
           float s_depth=(k2t21.at<float>(0)-k2t21.at<float>(2)*uj)/(Rxy1.at<float>(2)*uj-Rxy1.at<float>(0)+1e-10);
           float s_a_depth=s_depth*kR31K_xy1.at<float>(2)+kt31.at<float>(2);
           float u_a=(s_depth*kR31K_xy1.at<float>(0)+kt31.at<float>(0))/s_a_depth;
           float v_a=(s_depth*kR31K_xy1.at<float>(1)+kt31.at<float>(1))/s_a_depth;
            std::vector<float> photo_kf2,gradient_kf2,photo_kf1,gradient_kf1;
               photo_kf2.clear();
               photo_kf1.clear();
     std::vector<cv::Mat> n_point;
           for(int  j=-halfpath_size;j<halfpath_size+1;++j){
               for(int  i=-halfpath_size;i<halfpath_size+1;++i){
                          cv::Mat xppjj=(cv::Mat_<float>(3,1)<<i+u_a,j+v_a,1);
                          n_point.push_back(xppjj);
           }
           }
           for(size_t ii=0;ii<n_point.size();++ii){
               cv::Mat pjj=kR31K_.inv()*(s_a_depth*n_point[ii]-kt31);
               float u_p=pjj.at<float>(0)/pjj.at<float>(2);
               float v_p=pjj.at<float>(1)/pjj.at<float>(2);
               photo_kf1.push_back(bilinear<uchar>(kf1->im_,v_p,u_p));
               cv::Mat pjj2=kR32K_.inv()*(s_a_depth*n_point[ii]-kt32);
               float u_p2=pjj2.at<float>(0)/pjj2.at<float>(2);
               float v_p2=pjj2.at<float>(1)/pjj2.at<float>(2);
               photo_kf2.push_back(bilinear<uchar>(kf2->im_,v_p2,u_p2));
           }


            float sum_photo_kf2=0.0,sum_photo_kf1=0.0;
          for(auto &i:photo_kf1){
                sum_photo_kf1+=i;
          }
               sum_photo_kf1/=photo_kf1.size();

            for(auto &j:photo_kf2){
                sum_photo_kf2+=j;
             }
           sum_photo_kf2/=photo_kf2.size();
             float pjjfenzhi=0.0,pjjfenmu1=0.0,pjjfenmu2=0.0,gradientfenzhi=0.0,gradientfenmu1=0.0,gradientfenmu2=0.0;
            int census_filter=0;
              float nssd_score=0;
             float err=0.0;
             float err_support_weight=0,err_support_weight1=0;
            // cout<<"number="<<photo_kf1.size()<<endl;
             if(photo_kf1.size()!=photo_kf2.size()){cout<<"error photo_kf1 photo_kf2 is not same"<<endl;}

           for(size_t i=0;i<photo_kf1.size();i++){
             //  err_support_weight+=w_pq[i]*w1_pq[i]*abs(photo_kf1[i]-photo_kf2[i]);
              //err_support_weight1+=w_pq[i]*w1_pq[i];
                // cout<<"phot0kf1.size="<<photo_kf1.size()<<"photokf2.size="<<photo_kf2.size()<<endl;
               pjjfenzhi+=(photo_kf1[i]-sum_photo_kf1)*(photo_kf2[i]-sum_photo_kf2);
                    //census_filter+=pow(photo_kf1[i]-photo_kf2[i],2);//+pow(gradient_kf1[i]-gradient_kf2[i],2);
           pjjfenmu1+=pow(photo_kf1[i]-sum_photo_kf1,2);
                 pjjfenmu2+=pow(photo_kf2[i]-sum_photo_kf2,2);

            }

           float NCC=pjjfenzhi/(std::sqrt(pjjfenmu1*pjjfenmu2)+1e-10);//+gradientfenzhi/(std::sqrt(gradientfenmu1*gradientfenmu2)*THETA+1e-10);

          if(NCC>ncc_score){

            best_pixel = uj;
          ncc_score=NCC;
     }
        }
   if(ncc_score>0.85){
         //cout<<"old_err="<<ncc_score<<endl;
     // cout<<"bess_filter="<<best_census_filter<<endl;
      best_photometric_err = pixel - bilinear<uchar>(kf2->im_,-((a/b)*best_pixel+(c/b)),best_pixel);
      best_gradient_modulo_err= kf1->GradImg.at<float>(y,x)  - bilinear<float>( kf2->GradImg,-((a/b)*best_pixel+(c/b)),best_pixel);
        uj_plus = best_pixel + 1;
        vj_plus = -((a/b)*uj_plus + (c/b));
        uj_minus = best_pixel - 1;
       vj_minus = -((a/b)*uj_minus + (c/b));
      if(vj_plus<0||vj_minus<0||vj_plus>kf2->im_.rows||uj_minus>kf2->im_.cols){return;}
       pjju=best_pixel;
       pjjv=-((a/b)*pjju+(c/b));
        g = ((float)kf2->im_.at<uchar>(vj_plus, uj_plus) -(float) kf2->im_.at<uchar>(vj_minus, uj_minus))/2.0;
        q = ( kf2->GradImg.at<float>(vj_plus, uj_plus) -  kf2->GradImg.at<float>(vj_minus, uj_minus))/2.0;

        if(vj_plus<0||vj_minus<0||vj_plus>kf2->im_.rows||uj_minus>kf2->im_.cols)   //  if abs(a/b) is large,   a little step for uj, may produce a large change on vj. so there is a bug !!!  vj_plus may <0
        {
            std::cout <<"vj_minus="<<vj_minus<<"uj_minus="<<uj_minus<<endl;
          std::cout<<"vj_plus: "<<vj_plus<<" a/b: "<<a/b<<" c/b: "<<c/b<<std::endl;
          std::cout<<"best_pixel: "<<best_pixel<<" vj: "<<vj<<" old_err "<<old_err<<std::endl;
          std::cout<<"uj_plus: "<<uj_plus<<std::endl;
        }
        denomiator = (g*g + (1/THETA)*q*q);
      ustar = best_pixel + (g*best_photometric_err + (1/THETA)*q*best_gradient_modulo_err)/denomiator;
       ustar_var =2* kf2->I_stddev*kf2->I_stddev/denomiator;

        best_u = ustar;
        best_v =  -( (a/b)*best_u + (c/b) );
      //ComputeInvDepthHypothesis(kf1, kf2, umin,umax,ustar, ustar_var, a, b, c, dh,x,y);
    ComputeInvDepthHypothesis(kf1, kf2,umin,umax, ustar, 1, a, b, c, dh,x,y);
}
}
    //qwe
//qwer
 void ProbabilityMapping::IntraKeyFrameDepthChecking(cv::Mat& depth_map, cv::Mat& depth_sigma,const cv::Mat gradimg)
{
   //std::vector<std::vector<depthHo> > ho_new (depth_map.rows, std::vector<depthHo>(depth_map.cols, depthHo()) );
   cv::Mat depth_map_new = depth_map.clone();
   cv::Mat depth_sigma_new = depth_sigma.clone();

   for (int py = 2; py < (depth_map.rows - 2); py++)
   {

       for (int px = 2; px < (depth_map.cols - 2); px++)
       {

           if (depth_map.ptr<float>(py)[px] < 0.01)  // if  d ==0.0 : grow the reconstruction getting more density
           {
                  if(gradimg.ptr<float>(py)[px]<lambdaG) continue;
                  //search supported  by at least 2 of its 8 neighbours pixels
                  std::vector< depthHo > max_supported;

                  for( int  y = py - 1; y <= py+1; y++)
                    for( int  x = px - 1 ; x <= px+1; x++)
                    {
                        //uchar  *dp=&depth_map.data[y*depth_map.step+x];
                     //     uchar  *sp=&depth_sigma.data[y*depth_sigma.step+x];
                       if(depth_map.ptr<float>(y)[x]<0.01){continue;}//qwret
                      std::vector< depthHo>supported;
                      if(x == px && y == py) continue;
                      for (int nx = px - 1; nx <= px + 1; nx++)
                          for (int ny = py - 1; ny <= py + 1; ny++)
                          {
                            if((x == nx && y == ny) || (nx == px && ny == py))continue;
                               // uchar *dp1=&depth_map.data[ny*depth_map.step+nx];
                              // uchar  *sp1=&depth_sigma.data[ny*depth_sigma.step+nx];
                            if(depth_map.ptr<float>(ny)[nx]<0.01){continue;}//qwret
                            if(ChiTest(depth_map.ptr<float>(y)[x],depth_map.ptr<float>(ny)[nx],depth_sigma.ptr<float>(y)[x],depth_sigma.ptr<float>(ny)[nx]))
                            {
                                     depthHo depth;
                                     depth.depth= depth_map.ptr<float>(ny)[nx];
                                     depth.sigma = depth_sigma.ptr<float>(ny)[nx];
                                     supported.push_back(depth);
                            }
                          }

                      if(supported.size()>0  &&  supported.size() >= max_supported.size())   //  select the max supported neighbors
                      {

                          depthHo depth;
                          depth.depth=depth_map.ptr<float>(y)[x];
                          depth.sigma=depth_sigma.ptr<float>(y)[x];
                        supported.push_back(depth);   // push (y,x) itself
                       max_supported.clear();
                        max_supported = supported;
                   //   if(max_supported.size()!=0){ cout<<"max_supported.size"<<max_supported.size()<<endl;}

                      }
                    }

                  if(max_supported.size() > 1)//gaile
                  {

                    depthHo fusion1;
                    float min_sigma = 0;
                    GetFusion(max_supported, fusion1, &min_sigma);
                    depth_map_new.ptr<float>(py)[px]= fusion1.depth;
                    depth_sigma_new.ptr<float>(py)[px] = min_sigma;

                  }

           }
           else
           {
             std::vector<depthHo> compatible_neighbor_ho;

             depthHo dha,dhb;

             dha.depth=depth_map.ptr<float>(py)[px];
             dha.sigma=depth_sigma.ptr<float>(py)[px];
             for (int y = py - 1; y <= py + 1; y++)
             {
                 for (int x = px - 1; x <= px + 1; x++)
                 {

                   if (x == px && y == py) continue;

                   if( depth_map.ptr<float>(y)[x]> 0.01)
                   {
                     if(ChiTest(depth_map.ptr<float>(y)[x],depth_map.ptr<float>(py)[px],depth_sigma.ptr<float>(y)[x],depth_sigma.ptr<float>(py)[px]))
                     {
                       dhb.depth = depth_map.ptr<float>(y)[x];
                       dhb.sigma = depth_sigma.ptr<float>(y)[x];
                       compatible_neighbor_ho.push_back(dhb);
                     }

                   }
                 }
             }
             compatible_neighbor_ho.push_back(dha);  // dont forget itself.

             if (compatible_neighbor_ho.size() > 2)
             {
                 depthHo fusion;
                 float min_sigma = 0;
                 GetFusion(compatible_neighbor_ho, fusion, &min_sigma);

                 depth_map_new.ptr<float>(py)[px] = fusion.depth;
                 depth_sigma_new.ptr<float>(py)[px] = min_sigma;


             } else
             {
                 //ho_new[py][px].supported = false;   // outlier
                 depth_map_new.ptr<float>(py)[px] = 0.0;
                depth_sigma_new.ptr<float>(py)[px] = 0.0;

             }

           }
       }
   }

   depth_map = depth_map_new.clone();

   depth_sigma = depth_sigma_new.clone();



}


void ProbabilityMapping::IntraKeyFrameDepthChecking(std::vector<std::vector<depthHo> >& ho,int imrows, int imcols)
{

  std::vector<std::vector<depthHo> > ho_new (imrows, std::vector<depthHo>(imcols, depthHo()) );
  for (int py = 2; py < (imrows - 2); py++)
  {
      //std::cout<< "one row "<<std::endl;
      for (int px = 2; px < (imcols - 2); px++)
      {
          if (ho[py][px].supported == false)  // grow the reconstruction getting more density
          {

          }
          else
          {

            std::vector<depthHo> compatible_neighbor_ho;
            PixelNeighborSupport(ho, px, py, compatible_neighbor_ho);

            if (compatible_neighbor_ho.size() > 2)
            {
                // average depth of the retained pixels
                // set sigma to minimum of neighbor pixels

                depthHo fusion;
                float min_sigma = 0;
                GetFusion(compatible_neighbor_ho, fusion, &min_sigma);

                ho_new[py][px].depth = fusion.depth;
                ho_new[py][px].sigma = min_sigma;
                ho_new[py][px].supported = true;

            } else
            {
                ho_new[py][px].supported = false;   // outlier
            }

          }
      }
  }
  std::cout<< "end "<<std::endl;
  ho = ho_new;
}

//qwer
void ProbabilityMapping::InverseDepthHypothesisFusion(const std::vector<depthHo>& h, depthHo& dist) {
    dist.depth = 0;
    dist.sigma = 0;
    dist.supported = false;

    std::vector<depthHo> compatible_ho;
    std::vector<depthHo> compatible_ho_temp;
    float chi = 0;
    for (size_t a=0; a < h.size(); a++) {

        compatible_ho_temp.clear();
        for (size_t b=0; b < h.size(); b++)
        {  if(h[b].sigma==0||h[a].sigma==0){cout<<"qwer err"<<endl;}
          if (ChiTest(h[a], h[b], &chi))
            {compatible_ho_temp.push_back(h[b]);}// test if the hypotheses a and b are compatible
        }

        // test if hypothesis 'a' has the required support
       if (compatible_ho_temp.size() >= lambdaN)
        {
            compatible_ho.push_back(h[a]);
        }

    }


    if (compatible_ho.size() >lambdaN)
    {

       GetFusion(compatible_ho, dist, &chi);

    }
}

void ProbabilityMapping::InterKeyFrameDepthChecking(const cv::Mat& im, ORB_SLAM2::KeyFrame* currentKf, std::vector<std::vector<depthHo> >& h) {
    std::vector<ORB_SLAM2::KeyFrame*> neighbors;

    // option1: could just be the best covisibility keyframes
    neighbors = currentKf->GetBestCovisibilityKeyFrames(covisN);

    // option2: could be found in one of the LocalMapping SearchByXXX() methods
    //ORB_SLAM2::LocalMapping::SearchInNeighbors(); //mpCurrentKeyFrame->updateConnections()...AddConnection()...UpdateBestCovisibles()...
    //ORB_SLAM2::LocalMapping::mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(covisN); //mvpOrderedConnectedKeyFrames()


    for (int px = 0; px < im.rows; px++) {
        for (int py = 0; py < im.cols; py++) {
            if (h[px][py].supported == false) continue;

            float depthp = h[px][py].depth;
            // count of neighboring keyframes in which there is at least one compatible pixel
            int compatible_neighbor_keyframes_count = 0;
            // keep track of compatible pixels for the gauss-newton step
            std::vector<depthHo> compatible_pixels_by_frame[neighbors.size()];
            int n_compatible_pixels = 0;

            for(size_t j=0; j<neighbors.size(); j++) {
                ORB_SLAM2::KeyFrame* pKFj = neighbors[j];

                cv::Mat kj = pKFj->GetCalibrationMatrix();
                cv::Mat xp;
                GetXp(kj, px, py, &xp);

                cv::Mat rcwj = pKFj->GetRotation();
                cv::Mat tcwj = pKFj->GetTranslation();

                // Eq (12)
                // compute the projection matrix to map 3D point from original image to 2D point in neighbor keyframe
                cv::Mat temp;
                float denom1, denom2;

                temp = rcwj.row(2) * xp;
                denom1 = temp.at<float>(0,0);
                temp = depthp * tcwj.at<float>(2);
                denom2 = temp.at<float>(0,0);
                float depthj = depthp / (denom1 + denom2);

                cv::Mat xj2d = (kj * rcwj * (1 / depthp) * xp) + (kj * tcwj);
                float xj = xj2d.at<float>(0,0);
                float yj = xj2d.at<float>(1,0);

                std::vector<depthHo> compatible_pixels;
                // look in 4-neighborhood pixel p_j,n around xj for compatible inverse depth
                int pxn = floor(xj);
                int pyn = floor(yj);
                for (int nx = pxn-1; nx <= pxn + 1; nx++) {
                    for (int ny = pyn-1; ny < pyn + 1; ny++) {
                        if ((nx == ny) || ((nx - pxn) && (ny - pyn))) continue;
                        if (!h[nx][ny].supported) continue;
                        // Eq (13)
                        float depthjn = h[nx][ny].depth;
                        float sigmajn = h[nx][ny].sigma;
                        float test = pow((depthj - depthjn), 2) / pow(sigmajn, 2);
                        if (test < 3.84) {
                            compatible_pixels.push_back(h[nx][ny]);
                        }
                    }
                }
                compatible_pixels_by_frame[j] = compatible_pixels; // is this a memory leak?
                n_compatible_pixels += compatible_pixels.size();

                // at least one compatible pixel p_j,n must be found in at least lambdaN neighbor keyframes
                if (compatible_pixels.size()) {
                    compatible_neighbor_keyframes_count++;
                }
            } // for j = 0...neighbors.size()-1

            // don't retain the inverse depth distribution of this pixel if not enough support in neighbor keyframes
            if (compatible_neighbor_keyframes_count < lambdaN) {
                h[px][py].supported = false;
            } else {
                // gauss-newton step to minimize depth difference in all compatible pixels
                // need 1 iteration since depth propagation eq. is linear in depth
                float argmin = depthp;
                cv::Mat J(n_compatible_pixels, 1, CV_32F);
                cv::Mat R(n_compatible_pixels, 1, CV_32F);
                int n_compat_index = 0;
                int iter = 1;
                for (int k = 0; k < iter; k++) {
                    for (size_t j = 0; j < neighbors.size(); j++) {
                        cv::Mat xp;
                        GetXp(neighbors[j]->GetCalibrationMatrix(), px, py, &xp);

                        cv::Mat rji = neighbors[j]->GetRotation();
                        cv::Mat tji = neighbors[j]->GetTranslation();
                        for (size_t i = 0; i < compatible_pixels_by_frame[j].size(); i++) {
                            float ri = 0;
                            Equation14(compatible_pixels_by_frame[j][i], argmin, xp, rji, tji, &ri);
                            R.at<float>(n_compat_index, 0) = ri;

                            cv::Mat tempm = rji.row(2) * xp;
                            float tempf = tempm.at<float>(0,0);
                            depthHo tempdH = compatible_pixels_by_frame[j][i];
                            J.at<float>(n_compat_index, 0) = -1 * tempf / (pow(tempdH.depth, 2) * tempdH.sigma);

                            n_compat_index++;
                        }
                    }
                    cv::Mat temp = J.inv(cv::DECOMP_SVD) * R;
                    argmin = argmin - temp.at<float>(0,0);
                }
                h[px][py].depth = argmin;
            }
        } // for py = 0...im.cols-1
    } // for px = 0...im.rows-1
}
//qwer
void ProbabilityMapping::InterKeyFrameDepthChecking(ORB_SLAM2::KeyFrame* currentKf,int &biaozhi) {
std::vector<ORB_SLAM2::KeyFrame*> vector_next,neighbors1,neighbors;
// option1: could just be the best covisibility keyframes
vector_next = currentKf->GetBestCovisibilityKeyFrames(covisN);
for(size_t i=0;i<vector_next.size();++i){
    ORB_SLAM2::KeyFrame* kf=vector_next[i];
    if(kf->semidense_begin_inter){
        neighbors1.push_back(kf);
    }
}
if(neighbors1.size() < covisN) {
    biaozhi=0;
    return;}
neighbors=vector<ORB_SLAM2::KeyFrame*>(neighbors1.begin(),neighbors1.begin()+covisN);
//cout<<"neighbors="<<neighbors.size()<<endl;
// for each pixel of keyframe_i, project it onto each neighbor keyframe keyframe_j
// and propagate inverse depth

std::vector <cv::Mat> Rji,tji;
std::vector<int>vec_j;
for(size_t j=0; j<neighbors.size(); j++)
{
  ORB_SLAM2::KeyFrame* kf2 = neighbors[ j ];
 if(kf2->semidense_flag_==true){vec_j.push_back(j);}
  cv::Mat Rcw1 = currentKf->GetRotation();
  cv::Mat tcw1 = currentKf->GetTranslation();
  cv::Mat Rcw2 = kf2->GetRotation();
  cv::Mat tcw2 = kf2->GetTranslation();

  cv::Mat R21 = Rcw2*Rcw1.t();
  cv::Mat t21 = -Rcw2*Rcw1.t()*tcw1+tcw2;

  Rji.push_back(R21);
  tji.push_back(t21);

}

int cols = currentKf->im_.cols;
int rows = currentKf->im_.rows;
float fx = currentKf->fx;
float fy = currentKf->fy;
float cx = currentKf->cx;
float cy = currentKf->cy;
int remove_cnt(0),change_cnt(0),all_cnt(0);
for (int py = 2; py <rows-2; py++) {
    // uchar *currentKf1=&currentKf->depth_map_.data[py*currentKf->depth_map_.step];
    for (int px = 2; px < cols-2; px++) {

        if (currentKf->depth_map_.ptr<float>(py)[px] < 0.01) continue;   //  if d == 0.0  continue;
      all_cnt++;
        float depthp = currentKf->depth_map_.ptr<float>(py)[px] ;

        // count of neighboring keyframes in which there is at least one compatible pixel
        int compatible_neighbor_keyframes_count = 0;


        std::vector<ProbabilityMapping::three_pair> compatible_pixels_by_frames;
       // int n_compatible_pixels = 0;

       cv::Mat xp=(cv::Mat_<float>(3,1) << (px-cx)/fx, (py-cy)/fy,1.0);// inverse project.    if has distortion, this code shoud fix
        for(size_t j=0; j<neighbors.size(); j++) {

            ORB_SLAM2::KeyFrame* pKFj = neighbors[j];
            cv::Mat K = pKFj->GetCalibrationMatrix();


            cv::Mat temp = Rji[j] * xp /depthp + tji[j];
            cv::Mat Xj = K*temp;
            if(Xj.at<float>(2)<0){cout<<"inter_error"<<endl;}
            Xj = Xj/Xj.at<float>(2);   //   u = u'/z   ,  v = v'/z

            // Eq (12)
            // compute the projection matrix to map 3D point from original image to 2D point in neighbor keyframe
            temp = Rji[j].row(2) * xp;
            float denom1 = temp.at<float>(0,0);
            temp = depthp * tji[j].at<float>(2);
            float denom2 = temp.at<float>(0,0);
            float depthj = depthp / (denom1 + denom2);

            float xj = Xj.at<float>(0);
            float yj = Xj.at<float>(1);


            // look in 4-neighborhood pixel p_j,n around xj for compatible inverse depth

            if(xj < 1 || xj > cols-1 || yj<1 || yj>rows-1){
                //number.push_back(0);
                continue;
            }
            int x0 = (int)std::floor(xj);
            int y0 = (int )std::floor(yj);
            int x1 = x0 + 1;
            int y1 =  y0 + 1;

            std::vector<ProbabilityMapping::three_pair> compatible_pixels;
            compatible_pixels.clear();
             three_pair pjj;
           // float d = pKFj->depth_map_.at<float>(y0,x0);
             float d = pKFj->depth_map_.ptr<float>(y0)[x0];
            float sigma = pKFj->depth_sigma_.ptr<float>(y0)[x0];
            if(d>0.01)
            {
                if(sigma==0){cout<<"inter"<<endl;}
              //  cout<<"d-depthj="<<abs(d-depthj)<<endl;
                  float test = pow((depthj - d),2)/pow(sigma,2);
                 // cout<<"test="<<test<<endl;
                  if (test < 3.84) {
                        pjj.a=j;
                        pjj.b=d;
                        pjj.c=sigma;
                      compatible_pixels.push_back(pjj);
                  }
            }
           //  d = pKFj->depth_map_.at<float>(y1,x0);
             d = pKFj->depth_map_.ptr<float>(y1)[x0];
             sigma = pKFj->depth_sigma_.ptr<float>(y1)[x0];
            if(d>0.01)
            {//cout<<"d-depthj="<<abs(d-depthj)<<endl;
                 if(sigma==0){cout<<"inter"<<endl;}
                  float test = pow((depthj - d),2)/pow(sigma,2);
                 // cout<<"test="<<test<<endl;
                  if (test < 3.84) {
                      pjj.a=j;
                      pjj.b=d;
                      pjj.c=sigma;
                    compatible_pixels.push_back(pjj);
                  }
            }
          //  d = pKFj->depth_map_.at<float>(y0,x1);
            d = pKFj->depth_map_.ptr<float>(y0)[x1];
            sigma = pKFj->depth_sigma_.ptr<float>(y0)[x1];
           if(d>0.01)
           {//cout<<"d-depthj="<<abs(d-depthj)<<endl;
                if(sigma==0){cout<<"inter"<<endl;}
                 float test = pow((depthj - d),2)/pow(sigma,2);
                // cout<<"test="<<test<<endl;
                 if (test < 3.84) {
                     pjj.a=j;
                     pjj.b=d;
                     pjj.c=sigma;
                   compatible_pixels.push_back(pjj);
                 }
           }
          // d = pKFj->depth_map_.at<float>(y1,x1);
           d=pKFj->depth_map_.ptr<float>(y1)[x1];
           sigma=pKFj->depth_sigma_.ptr<float>(y1)[x1];
           //sigma = pKFj->depth_sigma_.at<float>(y1,x1);
          if(d>0.01)
          {//cout<<"d-depthj="<<abs(d-depthj)<<endl;
               if(sigma==0){cout<<"inter"<<endl;}
                float test = pow((depthj - d),2)/pow(sigma,2);
               // cout<<"test="<<test<<endl;
                if (test < 3.84) {
                    pjj.a=j;
                    pjj.b=d;
                    pjj.c=sigma;
                  compatible_pixels.push_back(pjj);
                }
          }

             if(compatible_pixels.size()){
             for(int i=0;i<compatible_pixels.size();i++){
              compatible_pixels_by_frames.push_back(compatible_pixels[i]);
             }
             }

            if (compatible_pixels.size()) {compatible_neighbor_keyframes_count++;}////

        } // for j = 0...neighbors.size()-1

        // don't retain the inverse depth distribution of this pixel if not enough support in neighbor keyframes
        if (compatible_neighbor_keyframes_count < lambdaN)
        {
           //currentKf->depth_map_.at<float>(py,px) = 0.0;
           currentKf->depth_map_.ptr<float>(py)[px]=0.0;
           // currentKf->depth_map_finally.ptr<float>(py)[px]=0.0;
            remove_cnt++;
        }

      else{
            vector<float> a,b,c;
            if(a.size()){cout<<"error a.size"<<endl;}
            for(size_t i=0;i<compatible_pixels_by_frames.size();++i){
                three_pair pjj_temp=compatible_pixels_by_frames[i];
                cv::Mat t=tji[pjj_temp.a];
                float t1=t.at<float>(2);
                float a1=1/pjj_temp.b-t1;
                a.push_back(a1);
                cv::Mat r=Rji[pjj_temp.a]*xp;
                float b1=r.at<float>(2);
                b.push_back(b1);
                float c1=pow(pjj_temp.b*pjj_temp.b,2)/pow(pjj_temp.c,2);
                c.push_back(c1);
            }


            if(a.size()!=b.size()||a.size()!=c.size()){cout<<"error"<<endl; }
            float abc_pjj=0,b2c_pjj=0;
            for(size_t m=0;m<a.size();++m){
                abc_pjj+=a[m]*b[m]*c[m];
                b2c_pjj+=b[m]*b[m]*c[m];
            }
            float inv_dp=b2c_pjj/abc_pjj;
          //  currentKf->depth_map_.at<float>(py,px) = inv_dp;
            currentKf->depth_map_.ptr<float>(py)[px]=inv_dp;
          //  currentKf->depth_map_finally.ptr<float>(py)[px]=inv_dp;
            change_cnt++;
            //qwe


        }

    }
} // for px = 0...im.rows-1
biaozhi=1;
cout<<"all_cnt="<<all_cnt<<"remove_cnt="<<remove_cnt<<"change_cnt="<<change_cnt<<endl;
}

void ProbabilityMapping::Equation14(depthHo& dHjn, float& depthp, cv::Mat& xp, cv::Mat& rji, cv::Mat& tji, float* res) {
    cv::Mat tempm = rji.row(2) * xp;
    float tempf = tempm.at<float>(0,0);
    float tji_z = tji.at<float>(2);
    *res = pow((dHjn.depth - (depthp * tempf) - tji_z) / (pow(dHjn.depth, 2) * dHjn.sigma), 1);
}


////////////////////////
// Utility functions
////////////////////////

void ProbabilityMapping::ComputeInvDepthHypothesis(ORB_SLAM2::KeyFrame* kf, ORB_SLAM2::KeyFrame* kf2, float umin ,float umax,float ustar, float ustar_var,
                                                   float a, float b, float c,ProbabilityMapping::depthHo *dh, int x,int y) {

  //float v_star=- ((a/b) * ustar + (c/b));
  float inv_pixel_depth =  0.0;
//if(ustar<umin){ cout<<"ustar-umin="<<ustar-umin<<endl;}
//if(ustar>umax){cout<<"ustar-umax="<<ustar-umax<<endl;}
  // equation 8 comput depth
  GetPixelDepth(ustar, x , y,kf, kf2,inv_pixel_depth);
 // if(inv_pixel_depth<0){inv_pixel_depth=1e-10;}
 // cout<<"inv_pixel_depth="<<inv_pixel_depth<<endl;
  // linear triangulation method
  // GetPixelDepth(ustar, pixel_y, x ,y,kf, kf2,inv_pixel_depth,dh);
  float ustar_min = ustar - sqrt(ustar_var);
  //int vstar_min = -((a/b)*ustar_min + (c/b));
//if(ustar_min<umin){ustar_min=umin;}

  float inv_depth_max = 0.0;
  GetPixelDepth(ustar_min,x,y,kf,kf2, inv_depth_max);
//  cout<<"inv_depth_max="<<inv_depth_max<<endl;
  //(inv_frame_rot[2]*corrected_image.at<float>(ustarcx_min ,vstarcx_min)-fx*inv_frame_rot[0]*corrected_image.at<float>(ujcx,vjcx))/(-transform_data[2][ustarcx_min][vstarcx_min]+fx*transform_data[0]);

  float ustar_max = ustar +  sqrt(ustar_var);

// if(ustar_max>umax) {   ustar_max=umax;}
  //int vstar_max = -((a/b)*ustar_max + (c/b));

  float inv_depth_min = 0.0;
  GetPixelDepth(ustar_max,x,y,kf, kf2,inv_depth_min);
//  cout<<"inv_depth_min="<<inv_depth_min<<endl;
  //(inv_frame_rot[2]*corrected_image.at<float>(ustarcx_max ,vstarcx_max)-fx*inv_frame_rot[0]*corrected_image.at<float>(ujcx,vjcx)/)/(-transform_data[2][ustarcx_max][vstarcx_max]+fx*transform_data[0]);

  // Equation 9
  float sigma_depth = cv::max(abs(inv_depth_max-inv_pixel_depth), abs(inv_depth_min-inv_pixel_depth));
  if(sigma_depth==0){cout<<"what"<<inv_depth_max<<"  "<<inv_pixel_depth<<"  "<<inv_depth_min<<"  "<<umin<<"  "<<umax<<"  "<<ustar<<"  "<<ustar_max<<"  "<<ustar_min<<ustar_var<<endl;}
//if(sigma_depth/inv_pixel_depth<0.1){
  dh->depth = inv_pixel_depth;
  dh->sigma = sigma_depth;
  dh->supported = true;//}
//cout<<"depth="<<dh->depth<<"dh.sigma="<<dh->sigma<<endl;
}

void ProbabilityMapping::GetGradientMagAndOri(const cv::Mat& image, cv::Mat* gradx, cv::Mat* grady, cv::Mat* mag, cv::Mat* ori) {

  *gradx = cv::Mat::zeros(image.rows, image.cols, CV_32F);
  *grady = cv::Mat::zeros(image.rows, image.cols, CV_32F);
  *mag =  cv::Mat::zeros(image.rows, image.cols, CV_32F);
  *ori = cv::Mat::zeros(image.rows, image.cols, CV_32F);

  //For built in version
  //cv::Scharr(image, *gradx, CV_32F, 1, 0);
  //cv::Scharr(image, *grady, CV_32F, 0, 1);

  cv::Scharr(image, *gradx, CV_32F, 1, 0, 1/32.0);
  cv::Scharr(image, *grady, CV_32F, 0, 1, 1/32.0);


  cv::magnitude(*gradx,*grady,*mag);
  cv::phase(*gradx,*grady,*ori,true);

}

//might be a good idea to store these when they get calculated during ORB-SLAM.
void ProbabilityMapping::GetInPlaneRotation(ORB_SLAM2::KeyFrame* k1, ORB_SLAM2::KeyFrame* k2, float* th) {
  std::vector<cv::KeyPoint> vKPU1 = k1->GetKeyPointsUn();
  DBoW2::FeatureVector vFeatVec1 = k1->GetFeatureVector();
  std::vector<ORB_SLAM2::MapPoint*> vMapPoints1 = k1->GetMapPointMatches();
  cv::Mat Descriptors1 = k1->GetDescriptors();

  std::vector<cv::KeyPoint> vKPU2 = k2->GetKeyPointsUn();
  DBoW2::FeatureVector vFeatVec2 = k2->GetFeatureVector();
  std::vector<ORB_SLAM2::MapPoint*> vMapPoints2 = k2 ->GetMapPointMatches();
  cv::Mat Descriptors2 = k2->GetDescriptors();

  std::vector<int> rotHist[histo_length];
  for(int i=0;i<histo_length;i++)
    rotHist[i].reserve(500);//DescriptorDistance

  const float factor = 1.0f;//histo_length;

  DBoW2::FeatureVector::iterator f1it = vFeatVec1.begin();
  DBoW2::FeatureVector::iterator f2it = vFeatVec2.begin();
  DBoW2::FeatureVector::iterator f1end = vFeatVec1.end();
  DBoW2::FeatureVector::iterator f2end = vFeatVec2.end();

  while(f1it != f1end && f2it != f2end) {
    if(f1it->first == f2it->first){
      for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++){
        size_t index1 = f1it->second[i1];

        ORB_SLAM2::MapPoint* pMP1 = vMapPoints1[index1];
        if(!pMP1)
          continue;
        if(pMP1->isBad())
          continue;

        cv::Mat d1 = Descriptors1.row(index1);

        int bestDist1 = INT_MAX;
        int bestIndex2 = -1;
        int bestDist2 = INT_MAX;
        size_t index2;
        for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++){
          index2 = f2it->second[i2];

          ORB_SLAM2::MapPoint* pMP2 = vMapPoints2[index2];
          if(!pMP2)
            continue;
          if(pMP2->isBad())
            continue;

          cv::Mat d2 = Descriptors2.row(index2);

          int dist = ORB_SLAM2::ORBmatcher::DescriptorDistance(d1,d2);

          if(dist<bestDist1){
            bestDist2 = bestDist1;
            bestDist1 = dist;
            bestIndex2 = index2;
          }
          else if(dist<bestDist2){
            bestDist2 = dist;
          }
        }
        if(bestDist1<th_low){
          if(static_cast<float>(bestDist1)<NNRATIO*static_cast<float>(bestDist2)){
            float rot = vKPU1[index1].angle - vKPU2[index2].angle;
            if(rot<0.0)
              rot+=360.0f;
            int bin = round(rot*factor);
            if(bin==histo_length)
              bin=0;
            rotHist[bin].push_back(index1);
          }
        }
      }
    }
  }
  //calculate the median angle
  size_t size = 0;
  for(int i=0;i<histo_length;i++)
    size += rotHist[i].size();

  size_t count = 0;
  for(int i=0;i<histo_length;i++) {
    for (size_t j=0; j < rotHist[i].size(); j++) {
        if (count==(size/2))
            *th = 360 * (float)(i) / histo_length;
        count++;
    }
  }
  //if(size % 2 == 0){
  //  *th = (rotHist[size/2 - 1] + rotHist[size/2])/2;
  //}
  //else{
  //  *th = rotHist[size/2];
  //}
}


void ProbabilityMapping::PixelNeighborSupport(std::vector<std::vector<depthHo> > H, int px, int py, std::vector<depthHo>& support) {
    support.clear();
    float chi = 0;
    for (int y = py - 1; y <= py + 1; y++) {
        for (int x = px - 1; x <= px + 1; x++) {

          if (x == px && y == py) continue;
          if(!H[y][x].supported) continue;

            if (ChiTest(H[y][x], H[py][px], &chi))
            {
                support.push_back(H[y][x]);
            }
        }
    }
    support.push_back(H[py][px]);  // dont forget itself.
}

void ProbabilityMapping::PixelNeighborNeighborSupport(std::vector<std::vector<depthHo> > H, int px, int py, std::vector<std::vector<depthHo> >& support) {
    support.clear();
    float chi = 0;
    for (int x = px - 1; x <= px + 1; x++) {
        for (int y = py - 1; y <= py + 1; y++) {
            if (x == px && y == py) continue;
            std::vector<depthHo> tempSupport;
            for (int nx = px - 1; nx <= px + 1; nx++) {
                for (int ny = py - 1; ny <= py + 1; ny++) {
                    if ((nx == px && ny == py) || (nx == x && ny == y)) continue;
                    if (ChiTest(H[x][y], H[nx][ny], &chi)) {
                        tempSupport.push_back(H[nx][ny]);
                    }
                }
            }
            support.push_back(tempSupport);
        }
    }
}

void ProbabilityMapping::GetIntensityGradient_D(const cv::Mat& ImGrad, float a, float b, float c, int px, float* q) {
    int uplusone = px + 1;
    int vplusone =- ((a/b)*uplusone + (c/b));
    int uminone = px - 1;
    int vminone = -((a/b)*uminone + (c/b));
    *q = (ImGrad.at<float>(uplusone,vplusone) - ImGrad.at<float>(uminone,vminone))/2;
}

void ProbabilityMapping::GetTR(ORB_SLAM2::KeyFrame* kf, cv::Mat* t, cv::Mat* r) {

    cv::Mat Rcw2 = kf->GetRotation();
    cv::Mat Rwc2 = Rcw2.t();
    cv::Mat tcw2 = kf->GetTranslation();
    cv::Mat Tcw2(3,4,CV_32F);
    Rcw2.copyTo(Tcw2.colRange(0,3));
    tcw2.copyTo(Tcw2.col(3));

    *t = Tcw2;
    *r = Rcw2;
}

void ProbabilityMapping::GetParameterization(const cv::Mat& F12, const int x, const int y, float& a, float& b, float& c) {
    // parameterization of the fundamental matrix (function of horizontal coordinate)
    // could probably use the opencv built in function instead
    a = x*F12.at<float>(0,0)+y*F12.at<float>(1,0)+F12.at<float>(2,0);
    b = x*F12.at<float>(0,1)+y*F12.at<float>(1,1)+F12.at<float>(2,1);
    c = x*F12.at<float>(0,2)+y*F12.at<float>(1,2)+F12.at<float>(2,2);
}

//Xp = K-1 * xp (below Equation 8)
// map 2D pixel coordinate to 3D point
void ProbabilityMapping::GetXp(const cv::Mat& k, int px, int py, cv::Mat* xp) {

    cv::Mat xp2d = cv::Mat(3,1,CV_32F);

    xp2d.at<float>(0,0) = px;
    xp2d.at<float>(1,0) = py;
    xp2d.at<float>(2,0) = 1;

    *xp = k.inv() * xp2d;
}

// Linear Triangulation Method
void ProbabilityMapping::GetPixelDepth(float uj, float vj, int px, int py, ORB_SLAM2::KeyFrame* kf,ORB_SLAM2::KeyFrame* kf2, float &p,ProbabilityMapping::depthHo *dh)
{

    float fx = kf->fx;
    float fy = kf->fy;
    float cx = kf->cx;
    float cy = kf->cy;

    cv::Mat R1w = kf->GetRotation();
    cv::Mat t1w = kf->GetTranslation();
    cv::Mat T1w(3,4,CV_32F);
    R1w.copyTo(T1w.colRange(0,3));  // 0,1,2 cols
    t1w.copyTo(T1w.col(3));

    cv::Mat R2w = kf2->GetRotation();
    cv::Mat t2w = kf2->GetTranslation();
    cv::Mat T2w(3,4,CV_32F);
    R2w.copyTo(T2w.colRange(0,3));
    t2w.copyTo(T2w.col(3));

    // inverse project.    if has distortion, this code shoud fix
    cv::Mat xn1 = (cv::Mat_<float>(3,1) << (px-cx)/fx, (py-cy)/fy,1.0);
    cv::Mat xn2 = (cv::Mat_<float>(3,1) << (uj-cx)/fx, (vj-cy)/fy, 1.0);

    cv::Mat A(4,4,CV_32F);
    A.row(0) = xn1.at<float>(0) * T1w.row(2) - T1w.row(0);
    A.row(1) = xn1.at<float>(1) * T1w.row(2) - T1w.row(1);
    A.row(2) = xn2.at<float>(0) * T2w.row(2) - T2w.row(0);
    A.row(3) = xn2.at<float>(1) * T2w.row(2) - T2w.row(1);

    cv::Mat w,u,vt;
    cv::SVD::compute(A,w,u,vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat pw = vt.row(3).t();
    if(pw.at<float>(3) == 0) return;

    cv::Mat pw_normalize = pw.rowRange(0,3) / pw.at<float>(3) ; // Point at world frame.
    //dh->Pw << pw_normalize.at<float>(0),pw_normalize.at<float>(1),pw_normalize.at<float>(2);
    //dh->Pw << pw_normalize.at<float>(0),pw_normalize.at<float>(1),1;

    //std::cout<<"linear method: "<<dh->Pw<<std::endl;

    cv::Mat x3Dt = pw_normalize.t();
    float z1 = R1w.row(2).dot(x3Dt)+t1w.at<float>(2);
    p = 1/z1;

}

// Equation (8)
void ProbabilityMapping::GetPixelDepth(float uj, int px, int py, ORB_SLAM2::KeyFrame* kf,ORB_SLAM2::KeyFrame* kf2, float &p) {

    float fx = kf->fx;
    float cx = kf->cx;
    float fy = kf->fy;
    float cy = kf->cy;
    cv::Mat R1w = kf->GetRotation();
    cv::Mat t1w = kf->GetTranslation();
    cv::Mat R2w = kf2->GetRotation();
    cv::Mat t2w = kf2->GetTranslation();
    cv::Mat R21=R2w*R1w.t();
    cv::Mat t21=-R2w*R1w.t()*t1w+t2w;
  //  if(uj>640||uj<0){
   //cout<<"error_uj="<<uj<<endl;}
   if(uj<0){uj=0;}
   if(uj>kf->im_.cols){uj=kf->im_.cols;}
    float ucx = uj - cx;
    cv::Mat xp=(cv::Mat_<float>(3,1) << (px-cx)/fx, (py-cy)/fy,1.0);// inverse project.    if has distortion, this code shoud fix

    //GetXp(kf->GetCalibrationMatrix(), px, py, &xp);

    cv::Mat temp = R21.row(2) * xp * ucx;
    float num1 = temp.at<float>(0,0);
    temp = fx * (R21.row(0) * xp);
    float num2 = temp.at<float>(0,0);
    float denom1 = -t21.at<float>(2) * ucx;
    float denom2 = fx * t21.at<float>(0);

     p = (num1 - num2) / (denom1 + denom2);
//cout<<"p1="<<p1<<endl;
}

void ProbabilityMapping::GetSearchRange(float& umin, float& umax, int px, int py,float mininvd,float maxinvd,
                                        ORB_SLAM2::KeyFrame* kf,ORB_SLAM2::KeyFrame* kf2)
{
  float fx = kf->fx;
  float cx = kf->cx;
  float fy = kf->fy;
  float cy = kf->cy;

  cv::Mat Rcw1 = kf->GetRotation();
  cv::Mat tcw1 = kf->GetTranslation();
  cv::Mat Rcw2 = kf2->GetRotation();
  cv::Mat tcw2 = kf2->GetTranslation();

  cv::Mat R21 = Rcw2*Rcw1.t();
  cv::Mat t21 = -Rcw2*Rcw1.t()*tcw1+tcw2;

  cv::Mat xp1=(cv::Mat_<float>(3,1) << (px-cx)/fx, (py-cy)/fy,1.0);  // inverse project.    if has distortion, this code shoud fix

  if(mininvd<0) mininvd = 1e-10;
  if(maxinvd<0)maxinvd=1e-10;
  cv::Mat xp2_max = R21*xp1/mininvd+t21;
  cv::Mat xp2_min = R21*xp1/maxinvd+t21;
if(xp2_max.at<float>(2)<0||xp2_min.at<float>(2)<0){cout<<"xp2_min,xp2_max="<<endl;}
  umin = fx*xp2_min.at<float>(0)/xp2_min.at<float>(2) + cx;
  umax = fx*xp2_max.at<float>(0)/xp2_max.at<float>(2) + cx;
if(umin>umax){
    float apjj=umax;
    umax=umin;
    umin=apjj;
}
  if(umin<1) umin = 1;
  if(umax<1) umax = 1;
  if(umin>kf->im_.cols-1 ) umin = kf->im_.cols-1;
  if(umax>kf->im_.cols-1)  umax = kf->im_.cols-1;
}

bool ProbabilityMapping::ChiTest(const depthHo& ha, const depthHo& hb, float* chi_val) {
    if(ha.sigma==0||hb.sigma==0){cout<<"chitest1_error"<<endl;}
    float num = (ha.depth - hb.depth)*(ha.depth - hb.depth);
    float chi_test = num / (ha.sigma*ha.sigma+1e-10) + num / (hb.sigma*hb.sigma+1e-10);
    if (chi_val)
        *chi_val = chi_test;
    return (chi_test < 5.99);  // 5.99 -> 95%
}

bool ProbabilityMapping::ChiTest(const float& a, const float& b, const float sigma_a,float sigma_b) {
   if(sigma_a==0||sigma_b==0){cout<<"chitest error"<<endl;}
    float num = (a - b)*(a - b);
    float chi_test = num / (sigma_a*sigma_a+1e-10) + num / (sigma_b*sigma_b+1e-10);
    return (chi_test < 5.99);  // 5.99 -> 95%
}
/*
void ProbabilityMapping::GetFusion(const std::vector<depthHo>& compatible_ho, depthHo* hypothesis, float* min_sigma) {
    hypothesis->depth = 0;
    hypothesis->sigma = 0;

    float temp_min_sigma = 100;
    float pjsj =0; // numerator
    float rsj =0; // denominator

    for (size_t j = 0; j < compatible_ho.size(); j++) {
        pjsj += compatible_ho[j].depth / pow(compatible_ho[j].sigma, 2);
        rsj += 1 / pow(compatible_ho[j].sigma, 2);
        if (pow(compatible_ho[j].sigma, 2) < pow(temp_min_sigma, 2)) {
            temp_min_sigma = compatible_ho[j].sigma;
        }
    }

    hypothesis->depth = pjsj / rsj;
    hypothesis->sigma = sqrt(1 / rsj);
    hypothesis->supported = true;

    if (min_sigma) {
        *min_sigma = temp_min_sigma;
    }
}
*/
void ProbabilityMapping::GetFusion(const std::vector<std::pair <float,float> > supported, float& depth, float& sigma)
{
    int t = supported.size();
    float pjsj =0; // numerator
    float rsj =0; // denominator
    for(size_t i = 0; i< t; i++)
    {
        if(supported[i].second==0){cout<<"getfusion error"<<endl;}
      pjsj += supported[i].first / pow(supported[i].second, 2);
      rsj += 1 / pow(supported[i].second, 2);
    }

    depth = pjsj / rsj;
    sigma = sqrt(1 / rsj);
}

 void ProbabilityMapping::GetFusion(const std::vector<depthHo>& compatible_ho, depthHo& hypothesis, float* min_sigma) {
    hypothesis.depth = 0;
    hypothesis.sigma = 0;
    hypothesis.supported=false;

    float temp_min_sigma = 100;
    float pjsj =0; // numerator
    float rsj =0; // denominator

    for (size_t j = 0; j < compatible_ho.size(); j++) {
        if(compatible_ho[j].sigma==0){cout<<"getfusion"<<endl;}
        pjsj += compatible_ho[j].depth / pow(compatible_ho[j].sigma, 2);
        rsj += 1 / pow(compatible_ho[j].sigma, 2);
        if (pow(compatible_ho[j].sigma, 2) < pow(temp_min_sigma, 2)) {
            temp_min_sigma = compatible_ho[j].sigma;
        }
    }
    hypothesis.depth = pjsj / rsj;
    hypothesis.sigma = sqrt(1 / rsj);
    hypothesis.supported = true;

    if (min_sigma) {
        *min_sigma = temp_min_sigma;
    }

}

cv::Mat ProbabilityMapping::ComputeFundamental( ORB_SLAM2::KeyFrame *&pKF1,  ORB_SLAM2::KeyFrame *&pKF2) {
    //qwer

    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = GetSkewSymmetricMatrix(t12);

    cv::Mat K1 = pKF1->GetCalibrationMatrix();
    cv::Mat K2 = pKF2->GetCalibrationMatrix();
    return K1.t().inv()*t12x*R12*K2.inv();

    //qwe




}

cv::Mat ProbabilityMapping::GetSkewSymmetricMatrix(const cv::Mat &v) {
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
                                  v.at<float>(2),               0,-v.at<float>(0),
                                 -v.at<float>(1),  v.at<float>(0),              0);
}
void  ProbabilityMapping::updataDepthFilter(const std::vector<depthHo> &depth_ho,depthHo &depth_temp){
    depthHo depth_pjj1,depth_pjj2;
    depth_pjj2=depth_ho[0];
    depth_temp.depth=depth_pjj2.depth;
    depth_temp.sigma=depth_pjj2.sigma;
    if(depth_ho.size()==1){
        //cout<<"depth_temp.sigma="<<depth_temp.sigma<<endl;
       if(depth_temp.sigma){
           depth_temp.supported=true;
       }
    }
    else{
    for(size_t i=1;i<depth_ho.size();++i){
        depth_pjj1=depth_ho[i];
        depth_temp.depth=(depth_pjj1.depth*pow(depth_temp.sigma,2)+depth_temp.depth*pow(depth_pjj1.sigma,2))/(pow(depth_pjj1.sigma,2)+pow(depth_temp.sigma,2));
        depth_temp.sigma=(pow(depth_pjj1.sigma,2)*pow(depth_temp.sigma,2))/(pow(depth_pjj1.sigma,2)+pow(depth_temp.sigma,2));
    }
    //cout<<"depth_temp.sigma="<<depth_temp.sigma<<endl;
    if(depth_temp.sigma){
        depth_temp.supported=true;
    }
    }
}
void ProbabilityMapping::updateseed(std::vector<depthHo>&depth_ho,seed& seed1,depthHo &depth_temp){
    for(size_t i=0;i<depth_ho.size();++i){
        depthHo invdepth=depth_ho[i];
    float norm_scale=std::sqrt(seed1.sigma2+invdepth.sigma*invdepth.sigma);
    if(std::isnan(norm_scale)){return;}
   // boost::random::normal_distribution<float> nd(seed1.mu,norm_scale);
    float s2=1/(1/seed1.sigma2+1/(invdepth.sigma*invdepth.sigma));
    float m=s2*(seed1.mu/seed1.sigma2+invdepth.depth/(invdepth.sigma*invdepth.sigma));
    float pdf=(1/std::sqrt(2*norm_scale*norm_scale*M_PI)*exp(-pow(invdepth.depth-seed1.mu,2)/(2*norm_scale*norm_scale)));
    float C1=seed1.a/(seed1.a+seed1.b)*pdf;
    float C2=seed1.b/(seed1.a+seed1.b)*1/seed1.d_range;
    float normalization_constant=C1+C2;
    C1=C1/normalization_constant;
    C2=C2/normalization_constant;
    float f=C1*(seed1.a+1)/(seed1.a+seed1.b+1)+C2*seed1.a/(seed1.a+seed1.b+1);
    float e=C1*(seed1.a+1)*(seed1.a+2)/((seed1.a+seed1.b+1)*(seed1.a+seed1.b+2))+C2*seed1.a*(seed1.a+1)/((seed1.a+seed1.b+1)*(seed1.a+seed1.b+2));
    float mu_new=C1*m+C2*seed1.mu;
    seed1.sigma2=C1*(s2+m*m)+C2*(seed1.sigma2+seed1.mu*seed1.mu)-mu_new*mu_new;
    seed1.mu=mu_new;
    seed1.a=(e-f)/(f-e/f);
    seed1.b=seed1.a*(1-f)/f;
    }
    float pai=seed1.a/(seed1.a+seed1.b);
    float fangcha=seed1.sigma2;
    if(pai<0.05){return;}

    if(pai>0.1&&fangcha<0.0001){
        depth_temp.depth=seed1.mu;
        depth_temp.sigma=std::sqrt(seed1.sigma2);
        depth_temp.supported=true;
    }
}
void ProbabilityMapping::getWarpMatrixAffine(ORB_SLAM2::KeyFrame*kf1,ORB_SLAM2::KeyFrame*kf2,int u,int v,float depth,int halfpatch_size,cv::Mat&A_kf2_kf1){
    cv::Mat R1w = kf1->GetRotation();
    cv::Mat t1w = kf1->GetTranslation();
    cv::Mat K1 = kf1->GetCalibrationMatrix();
    cv::Mat K2 = kf2->GetCalibrationMatrix();
    cv::Mat R2w = kf2->GetRotation();
    cv::Mat t2w = kf2->GetTranslation();
    cv::Mat px=(cv::Mat_<float>(3,1)<<u,v,1);
    cv::Mat k_inv_uv1=K1.inv()*px;
    cv::Mat pw=R1w.t()*(depth*k_inv_uv1-t1w);
    cv::Mat px_du=(cv::Mat_<float>(3,1)<<u+halfpatch_size,v,1);
    cv::Mat px_dv=(cv::Mat_<float>(3,1)<<u,v+halfpatch_size,1);
    cv::Mat pw_du=R1w.t()*(K1.inv()*px_du*depth-t1w);
    cv::Mat pw_dv=R1w.t()*(K1.inv()*px_dv*depth-t1w);
    cv::Mat px1=K2*(R2w*pw+t2w);
    px1=px1/px1.at<float>(2);
    //cout<<"px1"<<px1<<endl;
    cv::Mat px1_du=K2*(R2w*pw_du+t2w);
    px1_du=px1_du/px1_du.at<float>(2);
   // cout<<px1_du<<endl;
    cv::Mat px1_dv=K2*(R2w*pw_dv+t2w);
    px1_dv=px1_dv/px1_dv.at<float>(2);
      //cout<<px1_dv<<endl;
    //cv::Mat A;
      cv::Mat pjj1=(px1_du-px1)/halfpatch_size;
          cv::Mat pjj2=(px1_dv-px1)/halfpatch_size;
    //A_kf2_kf1=A.colRange(0,2).rowRange(0,2);
 A_kf2_kf1=(cv::Mat_<float>(2,2)<<pjj1.at<float>(0),pjj2.at<float>(0),pjj1.at<float>(1),pjj2.at<float>(1));
   //cout<<"A="<<A_kf2_kf1<<endl;
}
float ProbabilityMapping::rubang(float a,float b){
//   cout<<"a-b="<<a-b<<endl;
    if(abs(a-b)>180){
        return M_PI*(360-abs(a-b))/180;
    }
    else{
        return M_PI*abs(a-b)/180;
    }
}
cv::Mat ProbabilityMapping::kaifang(cv::Mat& a_pjj){
    cv::Mat z=cv::Mat::eye(3,3,CV_32F);
    cv::Mat y=a_pjj;
    for(size_t i=0;i<10;i++){
        cv::Mat y1=y;
        y=(y1+z.inv())/2;
        z=(z+y1.inv())/2;
       // cout<<"y="<<y<<endl;
    }
   // cout<<"//"<<endl;
    return y;
}
float ProbabilityMapping::fanwei(float a){
 float b=M_PI*abs(a)/180;
    if(b>M_PI){
        return (b-M_PI);
    }
    else{
        return b;
    }
}
cv::Mat ProbabilityMapping::getnormal(ORB_SLAM2::KeyFrame *kf,float x,float y){
    vector<size_t> vector_idx=kf->GetFeaturesInArea(x,y,5);
    cv::Mat normal_=(cv::Mat_<float>(3,1)<<0,0,1);
    vector<cv::Mat> vet_normal_temp;
    cv::Mat R=kf->GetRotation();
    cv::Mat sum_normal=(cv::Mat_<float>(3,1)<<0,0,0);
    for(auto &i:vector_idx){
        //cout<<"i="<<i<<endl;
        if(!i){continue;}
        ORB_SLAM2::MapPoint* mappoint=kf->GetMapPoint(i);
       //cout<<"////"<<endl;
      // cout<<"bool="<<mappoint->isBad()<<endl;
       if(mappoint==NULL){continue;}
       //cout<<mappoint->mnId<<endl;
        if(mappoint->isBad()){continue;}
             //  cout<<"/"<<endl;
        cv::Mat normal_temp=mappoint->GetNormal();
       // cout<<normal_temp<<endl;
        vet_normal_temp.push_back(normal_temp);
    }
   // cout<<"//"<<endl;
    if(vet_normal_temp.size()==0){return normal_;}

    else{
        for(auto&j:vet_normal_temp){
            sum_normal+=j;
        }
        sum_normal=sum_normal/vet_normal_temp.size();
       // cout<<R*sum_normal<<endl;
        return R*sum_normal;
    }
}
void ProbabilityMapping::Findpipei1(ORB_SLAM2::KeyFrame *kf1,std::vector<ORB_SLAM2::KeyFrame*> closestMatches,int x, int y,float pixel,float min_invdepth,
                                   float max_invdepth,int num_step,float &best_invdepth,float &best_sagma)
{
   std::vector<std::pair<float,int>>cha_pixel;
    float each_steplength=(max_invdepth-min_invdepth)/num_step;
    for(size_t i=0;i<num_step;++i){
        std::vector<std::pair<float,int>> depth_piexls_cha;
        float inv_depth_pjj=min_invdepth+each_steplength*i;
        for(size_t j=0;j<closestMatches.size();++j){
            ORB_SLAM2::KeyFrame*kf2=closestMatches[j];
            float pixel_huidu_gradient=-1;
            findpixel1(kf1,kf2,pixel,inv_depth_pjj,x,y,pixel_huidu_gradient);
            if(pixel_huidu_gradient!=-1){
            depth_piexls_cha.push_back(make_pair(pixel_huidu_gradient,i));
            }

        }
        if(depth_piexls_cha.size()==0){continue;}
        float sum_chazhi=0.0;
        for(auto &i:depth_piexls_cha){
            std::pair<float,int> pair_=i;
            sum_chazhi+=pair_.first;
        }
        cha_pixel.push_back(make_pair(sum_chazhi/depth_piexls_cha.size(),i));
    }
     if(cha_pixel.size()==0){return;}
         std::pair<float,int> pair_min_pixel=cha_pixel[0];
         float min_pixel=pair_min_pixel.first;
         int number=pair_min_pixel.second;
         for(size_t i=0;i<cha_pixel.size();++i){
             std::pair<float,int> pair_pjj=cha_pixel[i];
             float pixel_first=pair_pjj.first;
             int pixel_second=pair_pjj.second;
             if(pixel_first<min_pixel){
                 min_pixel=pixel_first;
                 number=pixel_second;

             }
         }


         best_invdepth=min_invdepth+each_steplength*number;
         best_sagma=each_steplength;
    }
void ProbabilityMapping::findpixel1(ORB_SLAM2::KeyFrame *kf1,ORB_SLAM2::KeyFrame *kf2,float pixel,float inv_depth,int x,int y,float &pixel_huidu_gradient){
    cv::Mat R1w = kf1->GetRotation();
    cv::Mat t1w = kf1->GetTranslation();
    cv::Mat R2w = kf2->GetRotation();
    cv::Mat t2w = kf2->GetTranslation();
    cv::Mat R21=R2w*R1w.t();
    cv::Mat t21=-R2w*R1w.t()*t1w+t2w;
    cv::Mat K2=kf2->GetCalibrationMatrix();
    cv::Mat K1 = kf1->GetCalibrationMatrix();
  //  cv::Mat xy1=(cv::Mat_<float>(3,1) << x, y,1.0);
    cv::Mat k2R21K1_=K2*R21*K1.inv();
    cv::Mat k2t21=K2*t21;
  //  cv::Mat Rxy1=k2R21K1_*xy1;
    float half_path_size=2;
    vector<cv::Mat> n_point;
    n_point.clear();;
    vector<float> photo_kf1,photo_kf2;
    photo_kf1.clear();
    photo_kf2.clear();
    for(size_t j=-half_path_size;j<=half_path_size;++j){
        for(size_t i=-half_path_size;i<=half_path_size;++i){
            cv::Mat pjj=(cv::Mat_<float>(3,1)<<x+i,j+y,1);
            n_point.push_back(pjj);
            photo_kf1.push_back(bilinear<uchar>(kf1->im_,j+y,i+x));
        }
    }
    if(n_point.size()==0){pixel_huidu_gradient=-1;
        return ;}
    for(size_t i=0;i<n_point.size();++i){
        cv::Mat uv=k2R21K1_*n_point[i]/inv_depth+k2t21;
        float u1=uv.at<float>(0)/uv.at<float>(2);
        float v1=uv.at<float>(1)/uv.at<float>(2);
        photo_kf2.push_back(bilinear<uchar>(kf2->im_,v1,u1));
    }
if(photo_kf1.size()!=photo_kf2.size()){cout<<"err with photo_kf.size photo_kf2.size"<<endl;}
float err=0;
for(size_t i=0;i<photo_kf1.size();++i){
    err+=pow(photo_kf1[i]/255-photo_kf2[i]/255,2);
}
if(err==0){pixel_huidu_gradient=-1;
return ;}
   pixel_huidu_gradient= err;
}
void ProbabilityMapping::findbestvecdepth(const std::vector<depthHo>& h, std::vector<depthHo>& dist) {
    std::vector<depthHo> compatible_ho;
    std::vector<depthHo> compatible_ho_temp;
    float chi = 0;

    for (size_t a=0; a < h.size(); a++) {

        compatible_ho_temp.clear();
        for (size_t b=0; b < h.size(); b++)
        {
          if (ChiTest(h[a], h[b], &chi))
            {compatible_ho_temp.push_back(h[b]);}// test if the hypotheses a and b are compatible
        }

        // test if hypothesis 'a' has the required support
        if (compatible_ho_temp.size() >= lambdaN )
        {
            compatible_ho.push_back(h[a]);
        }
    }

    // calculate the parameters of the inverse depth distribution by fusing hypotheses
    if (compatible_ho.size() >= lambdaN) {
          dist=compatible_ho;

    }
}
float ProbabilityMapping::geman_mcclure(float x){
    return (pow(x,2)/(pow(x,2)+4));
}

void ProbabilityMapping::IntraKeyFrameDepthChecking1(cv::Mat& depth_map, cv::Mat& depth_sigma,const cv::Mat gradimg)
{
  //std::vector<std::vector<depthHo> > ho_new (depth_map.rows, std::vector<depthHo>(depth_map.cols, depthHo()) );
  cv::Mat depth_map_new = depth_map.clone();
  cv::Mat depth_sigma_new = depth_sigma.clone();

  for (int py = 2; py < (depth_map.rows - 2); py++)
  {

      for (int px = 2; px < (depth_map.cols - 2); px++)
      {

          if (depth_map.ptr<float>(py)[px] < 0.01)  // if  d ==0.0 : grow the reconstruction getting more density
          {
                 if(gradimg.ptr<float>(py)[px]<lambdaG) continue;
                 //search supported  by at least 2 of its 8 neighbours pixels
                 std::vector< depthHo > max_supported;

                 for( int  y = py - 1; y <= py+1; y++)
                   for( int  x = px - 1 ; x <= px+1; x++)
                   {
                       //uchar  *dp=&depth_map.data[y*depth_map.step+x];
                    //     uchar  *sp=&depth_sigma.data[y*depth_sigma.step+x];
                      if(depth_map.ptr<float>(y)[x]<0.01){continue;}//qwret
                     std::vector< depthHo>supported;
                     if(x == px && y == py) continue;
                     for (int nx = px - 1; nx <= px + 1; nx++)
                         for (int ny = py - 1; ny <= py + 1; ny++)
                         {
                           if((x == nx && y == ny) || (nx == px && ny == py))continue;
                              // uchar *dp1=&depth_map.data[ny*depth_map.step+nx];
                             // uchar  *sp1=&depth_sigma.data[ny*depth_sigma.step+nx];
                           if(depth_map.ptr<float>(ny)[nx]<0.01){continue;}//qwret
                           if(ChiTest(depth_map.ptr<float>(y)[x],depth_map.ptr<float>(ny)[nx],depth_sigma.ptr<float>(y)[x],depth_sigma.ptr<float>(ny)[nx]))
                           {
                                    depthHo depth;
                                    depth.depth= depth_map.ptr<float>(ny)[nx];
                                    depth.sigma = depth_sigma.ptr<float>(ny)[nx];
                                    supported.push_back(depth);
                           }
                         }

                     if(supported.size()>0  &&  supported.size() >= max_supported.size())   //  select the max supported neighbors
                     {

                         depthHo depth;
                         depth.depth=depth_map.ptr<float>(y)[x];
                         depth.sigma=depth_sigma.ptr<float>(y)[x];
                       supported.push_back(depth);   // push (y,x) itself
                      max_supported.clear();
                       max_supported = supported;


                     }
                   }

                 if(max_supported.size() > 1)//gaile
                 {

                   depthHo fusion1;
                   float min_sigma = 0;
                   GetFusion(max_supported, fusion1, &min_sigma);
                   depth_map_new.ptr<float>(py)[px]= fusion1.depth;
                   depth_sigma_new.ptr<float>(py)[px] = min_sigma;

                 }

          }


          }

  }

  depth_map = depth_map_new.clone();

  depth_sigma = depth_sigma_new.clone();



}
/*void ProbabilityMapping:: fanwei_u_homography(const int &x,const int& y,float d,cv::Mat R21,cv::Mat t21,float& umin,float& umax,const int detau,ORB_SLAM2::KeyFrame*kf){
    cv::Mat pjj=(cv::Mat_<float>(3,1)<<x,y,1);
    cv::Mat p001=(cv::Mat_<float>(1,3)<<0,0,d);
    cv::Mat k=kf->GetCalibrationMatrix();
    cv::Mat h=R21+t21*p001;
    cv::Mat H=k*h*k.inv();
    cv::Mat uv1=H*pjj;
    float u=uv1.at<float>(0,0)/uv1.at<float>(0,2);

    if(u-detau>1&&u-detau<kf->im_.cols-1&&umin<u-detau){
        umin=u-detau;
    }
    if(u+detau<kf->im_.cols-1&&u+detau>1&&umax>u+detau){
        umax=u+detau;
    }
}
*/
