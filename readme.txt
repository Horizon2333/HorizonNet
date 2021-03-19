checkpoints：保存模型的默认文件夹
logs：保存控制台输出的默认文件夹
models：包含各种模型的定义代码
pretrained：用来保存预训练模型
runs：TensorBoard默认日志文件夹

result.jpg：保存loss和accuracy散点图

divide_train_val.py：划分训练集和测试集代码
Horizon.py：产生个性化输出
logger.py：将控制台输出映射到文件中
printmodel.py：使用torchsummary或者TensorBoard展示模型
train.py：训练和评估模型