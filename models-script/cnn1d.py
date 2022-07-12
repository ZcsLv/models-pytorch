import torch 
import torch.nn as nn
# 参考：https://www.jianshu.com/p/45a26d278473
# 参考(point)：https://blog.csdn.net/sunny_xsc1994/article/details/82969867
# 
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1,self).__init__()
        words_dim=30
        n_filters=10
        n_classes=3
        n_words=100
        ks=2
        # 卷积 激活  池化  线性  dropout和batchnorm未加
        #x的输入形式：(batches,sequence_words,words_dim)
        """net的初始化：#class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            #in_channels:  输入信号的通道，其实就相当于词向量的维度
            #out_channels：卷积产生的通道，相当于有多个卷积核，也就是filters的个数
            #kernel_size(int or tuple): 卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
            #stride=1：
        """
        self.conv1d=nn.Conv1d(in_channels=words_dim,out_channels=n_filters,kernel_size=ks,stride=1)
        self.relu=nn.ReLU()
        #参数如何设置？参考：https://blog.csdn.net/sunny_xsc1994/article/details/82969867
        self.maxpool=nn.MaxPool1d(kernel_size=n_words-ks+1,stride=1)
        #全连接
        self.linear=nn.Linear(in_features=n_filters*1,out_features=n_classes)
        # self.seq=nn.Sequential(self.conv1d,self.relu,self.maxpool,self.linear)
    def forward(self,x):
        x=x.permute(0,2,1)
        x=self.conv1d(x)    # (batches,n_filters,n_words-ks/strides+1)  在卷积的过程中，词向量维度已经消掉了
        x=self.relu(x)      # (batches,n_filters,n_words-ks/strides+1)
        x=self.maxpool(x)   # (batches,n_filters,1)
        # 把x做一下cat,也就是把max之后的结果进行concat:(batches,n_filters,1)-->(batches,n_filters*1)
        x=x.view(x.size()[0],x.size()[1]) # (batches,n_filters)
        x=self.linear(x)    # (batches,n_filters)-->(batches,n_classes)
        return x
# 测试
# net1=CNN1()
# sequence_words=100
# words_dim=30
# batches=5
# input=torch.randn(batches,sequence_words,words_dim)
# output=net1(input)
# output.shape