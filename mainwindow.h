#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include<opencv2\opencv.hpp>
#include<string>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include "opencv2\face.hpp"
#include<iostream>
#include<QMessageBox>
#include<qdebug>
using namespace std;
using namespace cv;

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    bool loadFaceDectXml();//加载人脸检测xml
    void showCamera();//显示图像
    Mat detectionFace(Mat img);//检测人脸
    Mat grayFace(Mat input, vector<Rect>* faces);//返回脸部灰度图
    int updateLabels();//更新标签
    int trainface(Mat img);//人脸训练
    Mat testvideo(Mat img);
    bool addPic(Mat img);//抓拍一张图片
    void catchOver();//抓拍完毕

public slots:
    void on_btn_opencamera_clicked();
    void on_btn_trainface_clicked();
    void on_btn_testface_clicked();
    void on_btn_catchonce_clicked();
    void on_btn_catchover_clicked();
private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
