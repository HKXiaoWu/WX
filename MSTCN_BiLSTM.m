function [MSTCN_BiLSTM_ConcateAdd, netMSTCN_BiLSTM_ConcateAdd, PredY] = MultiScaleTCN_BiLSTM_concatentionLayer_Conv_additionLayer(numFeatures, numClasses, TrainX, TestX, TrainY, TestY, classes_new, InverseWeights, DilationAll, FilterSize, NumFilter, miniBatchSize, SpatialDropoutFactor, numHiddenUnits)
%MultiScaleTCNwithoutSE 此处显示有关此函数的摘要
%   此处显示详细说明
%   numFeatures: 模型输入特征变量维度   numClasses：类标签种类
%   TrainX: 训练集特征输入    TrainY: 训练集输出，类别标签
%   TestX: 测试集特征输入    TestY: 测试集输出，类别标签
%   classes_new: 类别标签 categories类型  InverseWeights: 反频率权重， 解决类不平衡
%   DilationAll: 全部DCC模块的空洞卷积因子

%   模型超参数：   FilterSize:过滤器大小 NumFilters:过滤器数量 miniBatchSize:最小批量大小   
%   dropoutFactor: spatialDropoutLayer的超参数

%% 网络结构

maxEpochs = 100 * (miniBatchSize/16);   % 最大迭代轮次
%iterPerEpoch = floor(length(TrainX)/miniBatchSize);   % 每个轮次的迭代次数

FilterSize1 = FilterSize(1,1);
FilterSize2 = FilterSize(1,2);

% 时间序列输入层
layersInput = sequenceInputLayer(numFeatures, Normalization="zscore",Name="Input", MinLength=180);   % 现场原始数据集: Zscore归一化

lgraphMultiScale_TCN = layerGraph(layersInput); 
inputName = layersInput.Name;

% TCN第一个尺度DCC
dilationFactorMutli1 = DilationAll(1);
layersTCN_Multi1 = [...
        % 第一个Dialted Causal Convolution
        convolution1dLayer(FilterSize1, NumFilter, DilationFactor=dilationFactorMutli1, Padding="causal", Name="Multi1_conv1")
        batchNormalizationLayer  
        geluLayer
        spatialDropoutLayer(SpatialDropoutFactor,'Name','Multi1_S1')
        
         % 第二个Dialted Causal Convolution
        convolution1dLayer(FilterSize2, NumFilter, DilationFactor=dilationFactorMutli1, Padding="causal", Name="Multi1_conv2")
        batchNormalizationLayer
        geluLayer
        spatialDropoutLayer(SpatialDropoutFactor,'Name','Multi1_S2')   
        ];


% TCN第二个尺度DCC
dilationFactorMutli2 = DilationAll(2);
layersTCN_Multi2 = [...
        % 第一个Dialted Causal Convolution
        convolution1dLayer(FilterSize1, NumFilter, DilationFactor=dilationFactorMutli2, Padding="causal", Name="Multi2_conv1")
        batchNormalizationLayer 
        geluLayer
        spatialDropoutLayer(SpatialDropoutFactor,'Name','Multi2_S1')
        
         % 第二个Dialted Causal Convolution
        convolution1dLayer(FilterSize2, NumFilter, DilationFactor=dilationFactorMutli2, Padding="causal", Name="Multi2_conv2")
        batchNormalizationLayer
        geluLayer
        spatialDropoutLayer(SpatialDropoutFactor,'Name','Multi2_S2') 
        ];

% TCN第三个尺度DCC
dilationFactorMutli3 = DilationAll(3);
layersTCN_Multi3 = [...
        % 第一个Dialted Causal Convolution
        convolution1dLayer(FilterSize1, NumFilter, DilationFactor=dilationFactorMutli3, Padding="causal", Name="Multi3_conv1")
        batchNormalizationLayer 
        geluLayer
        spatialDropoutLayer(SpatialDropoutFactor,'Name','Multi3_S1')
        
         % 第二个Dialted Causal Convolution
        convolution1dLayer(FilterSize2, NumFilter, DilationFactor=dilationFactorMutli3, Padding="causal", Name="Multi3_conv2")
        batchNormalizationLayer
        geluLayer
        spatialDropoutLayer(SpatialDropoutFactor,'Name','Multi3_S2')   
        ];


% 模型网络连接: 多维时间序列输入层、特征提取层、分类层
% 连接第一个尺度DCC
lgraphMultiScale_TCN = addLayers(lgraphMultiScale_TCN, layersTCN_Multi1);
lgraphMultiScale_TCN = connectLayers(lgraphMultiScale_TCN, inputName, 'Multi1_conv1');

% 连接第二个尺度DCC
lgraphMultiScale_TCN = addLayers(lgraphMultiScale_TCN, layersTCN_Multi2);
lgraphMultiScale_TCN = connectLayers(lgraphMultiScale_TCN, inputName, 'Multi2_conv1');

% 连接第三个尺度DCC
lgraphMultiScale_TCN = addLayers(lgraphMultiScale_TCN, layersTCN_Multi3);
lgraphMultiScale_TCN = connectLayers(lgraphMultiScale_TCN, inputName, 'Multi3_conv1');


%% 添加TCN结构中额外卷积层
layerCNN = convolution1dLayer(1, NumFilter, "Name","convSkip1");
lgraphMultiScale_TCN = addLayers(lgraphMultiScale_TCN, layerCNN);
lgraphMultiScale_TCN = connectLayers(lgraphMultiScale_TCN, inputName, "convSkip1");

% 层输出连接：多尺度特征融合
ConcatMultiScale = concatenationLayer(1,3, "Name","con_Mul");  
lgraphMultiScale_TCN = addLayers(lgraphMultiScale_TCN, ConcatMultiScale);
lgraphMultiScale_TCN = connectLayers(lgraphMultiScale_TCN, "Multi1_S2", "con_Mul/in1");
lgraphMultiScale_TCN = connectLayers(lgraphMultiScale_TCN, "Multi2_S2", "con_Mul/in2");
lgraphMultiScale_TCN = connectLayers(lgraphMultiScale_TCN, "Multi3_S2", "con_Mul/in3");

%% 采用1DCNN对多尺度特征进行通道降维，保证与额外的1DCNN残差连接的特征通道数一致
FeatureChannelCNN = convolution1dLayer(1, NumFilter, "Name","cnn1");  
lgraphMultiScale_TCN = addLayers(lgraphMultiScale_TCN, FeatureChannelCNN);
lgraphMultiScale_TCN = connectLayers(lgraphMultiScale_TCN, "con_Mul", "cnn1");

%% 层输出连接：残差连接
AdditionConnect = additionLayer(2,"Name",'addOutput');
lgraphMultiScale_TCN = addLayers(lgraphMultiScale_TCN, AdditionConnect);
lgraphMultiScale_TCN = connectLayers(lgraphMultiScale_TCN, "cnn1", "addOutput/in1");
lgraphMultiScale_TCN = connectLayers(lgraphMultiScale_TCN, "convSkip1", "addOutput/in2");

%% 分类输出层
layersClassification = [...
    globalMaxPooling1dLayer("Name", "GMP")
    bilstmLayer(numHiddenUnits,"OutputMode","sequence", 'Name','bilstm')
    fullyConnectedLayer(numClasses, "Name","FC")
    softmaxLayer("Name","SM")
    classificationLayer(Classes=classes_new, ClassWeights= InverseWeights)];
    % classficationLayer采用反频率权重技术来解决类不平衡问题

% 模型整体结构
lgraphMultiScale_TCN = addLayers(lgraphMultiScale_TCN, layersClassification);
lgraphMultiScale_TCN = connectLayers(lgraphMultiScale_TCN, "addOutput", "GMP");

MSTCN_BiLSTM_ConcateAdd = lgraphMultiScale_TCN;

%%  定义网络训练选项
MultiScaleTCN_options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'gpu', ...
    'L2Regularization', 1e-3, ...
    MiniBatchSize=miniBatchSize, ...
    MaxEpochs=maxEpochs, ...
    SequencePaddingDirection="left", ...
    ValidationData={TestX, TestY}, ...   % 测试集验证训练过程
    OutputNetwork="best-validation-loss", ...
    InitialLearnRate = 0.005, ...   % 原来设为0.005
    LearnRateSchedule = 'piecewise', ...
    Plots="training-progress", ...
    Verbose=0);

%% 训练网络
[netMSTCN_BiLSTM_ConcateAdd, ~] = trainNetwork(TrainX, TrainY, MSTCN_BiLSTM_ConcateAdd, MultiScaleTCN_options);

%% 测试网络
PredY = classify(netMSTCN_BiLSTM_ConcateAdd, TestX, ...
    MiniBatchSize = miniBatchSize, ...
    SequencePaddingDirection="left");

end
