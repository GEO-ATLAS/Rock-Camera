# 注意力机制
## 空间注意力机制
1）输入图像序列，经过2D卷积input_dim->output_dim，生成feature_map
2）对输入图像进行transformer，self-Attention操作，生成spatial_weight_map
3）卷积特征与spatial_weight_map进行dot操作得到weight_feature_map
4）使用weight_feature_map,传入多层堆叠的ConvLSTM,进行图像预测


## 时间轴注意力机制

输入帧，根据沿序列添加自注意力

