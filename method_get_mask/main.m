clear
close all

%------read images---------
%[imagename1, imagepath1]=uigetfile('./images/*.jpg;*.bmp;*.png;*.tif;*.tiff;*.pgm;*.gif','Please choose the first input image');
%A=imread(strcat(imagepath1,imagename1));    
%[imagename2, imagepath2]=uigetfile('./images/*.jpg;*.bmp;*.png;*.tif;*.tiff;*.pgm;*.gif','Please choose the second input image');
%B=imread(strcat(imagepath2,imagename2));  

folder='D:\数据集\多聚焦新数据集\训练集_测试集\测试集\';
for i=1:30
    A  = imread(strcat(folder,'2\crop\',num2str(i),'.jpg'));
    B  = imread(strcat(folder,'1\crop\',num2str(i),'.jpg'));

    A = double(A)/255;
    B = double(B)/255;

    if size(A)~=size(B)
        error('two images are not the same size.');
    end

    if size(A,3)>1
        A_gray=rgb2gray(A);
        B_gray=rgb2gray(B);
    else
        A_gray=A;
        B_gray=B;
    end

    [high, with] = size(A_gray);
    tic;
    % Parameters for Guided Filter 
    r =5; eps = 0.3;
    w =7;

    h = fspecial('average', [w w]);
    averA = imfilter(A_gray,h,'replicate');

    averB = imfilter(B_gray,h,'replicate');

    smA = abs(A_gray - averA);
    smB = abs(B_gray - averB);
    gsmA = guidedfilter_LKH(A_gray,smA, r, eps);
    gsmB = guidedfilter_LKH(B_gray,smB, r, eps);
    X= gsmA > gsmB;
    
    wmap0 = double(smA > smB);
    wmap = double(gsmA > gsmB);


%mask生成地址
    figure; imshow(wmap);
    imwrite(wmap,strcat('D:\数据集\多聚焦新数据集\训练集_测试集\测试集\M_I_crop\ll\',num2str(i),'.jpg'));
end
