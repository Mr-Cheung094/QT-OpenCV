#include "mainwindow.h"
#include "ui_mainwindow.h"

//人脸检测xml
//string Path = "D:\\Virtual box\\Share-File\\build-face-demo-Desktop_Qt_5_15_2_MinGW_64_bit-Debug\\debug\\resources\\haarcascade_frontalface_alt.xml";
string Path = ".\\haarcascade_frontalface_alt.xml";

string Pathwrite = ".\\train.xml";
Ptr<face::FaceRecognizer> recognizer;//FaceRecognizer：人脸识别、训练、跟踪类
CascadeClassifier Classifier;//定义人脸分类器
int  btrainface;//开始训练标志
vector<cv::Mat> trainingImage;//正在训练的人脸照片
vector<int> labels;//人脸对应的标签
int a;
bool flagbool;
int labelnumber;


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    btrainface = 0;//人脸检测模式
    a = 0;
    flagbool = true;
    labelnumber=0;
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btn_trainface_clicked()
{
    btrainface = 1;//开始训练人脸模型
    flagbool = false;
}
void MainWindow::on_btn_testface_clicked()
{
    btrainface = 2;//根据输入的模型进行检测人脸是否匹配
}
void MainWindow::on_btn_catchonce_clicked()
{
    btrainface = 3;//抓拍一张图片
}
void MainWindow::on_btn_catchover_clicked()
{
    btrainface = 4;//抓拍图片完毕
}

//点击“打开摄像头”按钮触发槽函数
void MainWindow::on_btn_opencamera_clicked()
{
    bool errorxml=loadFaceDectXml();//加载xml文件
    if (errorxml==true)
    {
        return;
    }
    btrainface = 0;
    showCamera();//图像显示
}

//判断人脸识别xml是否加载成功，成功返回0，失败返回-1
bool MainWindow::loadFaceDectXml()
{
    Classifier.load(Path);//加载级联分级器
    if (!Classifier.load(Path))  //加载训练文件
    {
        QMessageBox::warning(this, "警告", "加载人脸检测XML失败！");
        return 1;
    }
    else
    {
        //xml加载成功
        recognizer = face::LBPHFaceRecognizer::create(1, 8, 8, 8, 200.);//opencv自带LBPHFaceRecognizer算法
        return 0;
    }
}
//摄像头图像显示
void MainWindow::showCamera()
{
    VideoCapture cap(1);//打开电脑自带摄像头
    cap.set(CAP_PROP_EXPOSURE, 0);
    Mat camera;
    vector<Rect> faces;
    if (!cap.isOpened())
    {
        QMessageBox::warning(this, "警告", "摄像头无法正常打开！");
        return;
    }
    while (1)//开启显示摄像头内容
    {
        cap >> camera;
        //flip(camera, camera, 1);//翻转
        if (btrainface ==0)//默认
        {
            camera = detectionFace(camera);
        }
        else if(btrainface == 1)//训练模型
        {
            trainface(camera);
            btrainface =0;
        }
        else if (btrainface == 2)//人脸识别
        {
            if (flagbool==true)
            {
                recognizer->read(Pathwrite);
                flagbool = false;
            }

            camera = testvideo(camera);
        }
        else if (btrainface == 3)//抓拍一张图片
        {
            addPic(camera);
            btrainface = 0;
        }
        else if (btrainface == 4)//抓拍结束
        {
            catchOver();
            btrainface = 0;
        }
        imshow("Camera", camera);
        waitKey(30);
        if (waitKey(30) == 27)
        {
            cap.release();
            break;
        }
    }

}

//播放图像
Mat MainWindow::detectionFace(Mat img)
{
    vector<Rect> faces;
    Mat graysum = grayFace(img, &faces);//返回纯黑色（没有检测到人脸）或者灰度图（检测到有人脸）
    //Scalar he = sum(graysum);//检测到的人脸的数量和
    //if (he[0] == 0)
    if (faces.size()==0)
    {
        cv::resize(img, img, Size(), 1,1, INTER_AREA);
        return img;
    }
    //检测到人脸按照下面来显示
    cv::resize(img, img, Size(), 1, 1, INTER_AREA);
    rectangle(img, faces[0], Scalar(255, 255, 255), 2);

    return img;
}
//根据是否检测到人脸进行区分输出图像
Mat MainWindow::grayFace(Mat input, vector<Rect>* faces)
{
    Mat result;
    cv::resize(input, input, Size(), 1,1, INTER_AREA);
    cvtColor(input, result, COLOR_BGR2GRAY);//转灰度图
    equalizeHist(result, result);//灰度图二值化
    Mat zero = Mat::zeros(result.rows, result.cols, CV_8UC1);//纯黑色图片
    Classifier.detectMultiScale(result, *faces,1.1, 2, 0 | CV_HAL_CMP_GE, Size(100, 100), Size(200, 200));

    if ((*faces).size() == 0)//没有人脸返回纯黑色
        //QMessageBox::warning(this, "警告！", "无效！请重新抓拍！");
        return zero;

    Mat grayc = result.clone();//检测到人脸的情况
    grayc = grayc((*faces)[0]);

    return grayc;
}
//抓拍一张图片
bool  MainWindow::addPic(Mat img)
{
    Mat frame;
    vector<Rect> faces;
    int inputlabel;
    frame = grayFace(img, &faces);
    if (faces.size()== 0)
    {
        QMessageBox::warning(this, "警告！", "没有检测到人脸！请重新抓拍！");
        return 0;//失败
    }
    else
    {
        inputlabel = updateLabels();
        trainingImage.push_back(frame);
        labels.push_back(inputlabel);
        return 1;

    }
}
//抓拍完毕
void MainWindow::catchOver()
{
    labelnumber++;
    a++;
}
//开始训练人脸模型
int MainWindow::trainface(Mat img)
{
    recognizer->train(trainingImage, labels);
    recognizer->write(Pathwrite);
    QMessageBox::information(this, "提示", "人脸训练完成！");
    return 0;
}

int MainWindow::updateLabels()
{
    int inputlabel;

    inputlabel =labelnumber;
    return inputlabel;
}

Mat MainWindow::testvideo(Mat img)
{
    int predictedLabel = -1;
    double confidence = 0.0;
    vector<Rect> faces;
    Mat zero = Mat::zeros(img.rows, img.cols, CV_8UC1);

    Mat grayt = grayFace(img, &faces);
    if (faces.size() == 0)
    {
        //qDebug() << "图片中无人脸" << endl;
        cv::resize(img, img, Size(), 1, 1, INTER_AREA);
        return img;
    }
    //	qDebug() << "图片中you人脸" << endl;
    cv::resize(img, img, Size(), 1, 1, INTER_AREA);
    rectangle(img, faces[0], Scalar(255, 255, 255), 2);

    recognizer->predict(grayt,     // face image
                        predictedLabel, // predicted label of this image
                        confidence);    // confidence of the prediction
    string name;

    if (confidence < 80)
    {
        name = to_string(predictedLabel);
    }
    else {
        name = "unKnown!";
        confidence = 0;
    }
    putText(img, name, Point2d(faces[0].x, faces[0].y+30), \
            FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2, LINE_AA);

    return img;
}

