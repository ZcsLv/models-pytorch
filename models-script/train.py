import torch
import torch.nn as nn
import logging
import time
import torch.nn.functional as F

def train(train_iter,dev_iter,test_iter,args,model):
    start_time = time.time()
    steps=0
    logging.info("Training start time:",start_time)
    # 模型、数据放到gpu上。
    # 方法1）：.cuda()   方法2）：.to(device) device是指定好的
    model.cuda()
    # 指定模型的  方式
    model.train()
    # step 1:定义优化器
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    # step 2:
    for epoch in range(args.epoches):
        logging.info("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, args.epochs))
        # step 3: 读取batch个数据
        for batch in train_iter:
            # 1) 准备数据
            text,label=batch.text,batch.label
            text,label=text.cuda(),label.cuda()
            # 2) 定义损失函数
            lossf=nn.CrossEntropyLoss()
            # 3）梯度归零
            optimizer.zero_grad()
            # 4) 
            logit=model(text)
            # 5) 计算loss
            loss=lossf(logit,label)
            # 6) 
            loss.backward()
            optimizer.step()
            steps+=1
            if steps % args.log_intervel==0:
                train_size = len(train_iter.dataset)
                corrects = (torch.max(logit, 1)[1].view(label.size()).data == label.data).sum()
                accuracy = float(corrects)/batch.batch_size * 100.0
                #是用loss 还是用loss.item?
                logging.info("\n## train_loss {} , train_acc {} ##".format(loss.item(), accuracy))
            if steps % args.eval_interval==0:
                avg_loss,accuracy=eval(args,model,dev_iter)
                logging.info("\n## dev_loss {} , dev_acc {} ##".format(avg_loss, accuracy))
def eval(args,model,eval_iter):
    model.eval()
    loss_avg,corrects=0,0
    size=len(eval_iter)
    for batch in eval_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        logit=model(feature)

        loss=F.cross_entropy(logit,target)
        loss_avg+=loss
        corrects=(torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    size = len(eval_iter.dataset)
    avg_loss = loss.item()/size
    accuracy = 100.0 * float(corrects)/size
    return avg_loss,accuracy
def test(args,model,test_iter):
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    # start_time = time.time()
    test_loss, test_acc= eval(args, model, test_iter)
    return test_loss,test_acc
def parse_args():
    parser = argparse.ArgumentParser(description="Run LA_HCN.")

    # hyper-para for datasets
    parser.add_argument('--dataname', type=str, default='enron_2', help="training data.")
    parser.add_argument('--training_data_file', type=str, default='data/large/train/train_almg.json', help="path to training data.")
    parser.add_argument('--validation_data_file', type=str, default='data/large/val/val_almg.json', help="path to validation data.")
    parser.add_argument('--num_classes_list', type=str, default="8,129", help="Number of labels list (depends on the task)")
    parser.add_argument('--glove_file', type=str, default="data/glove6b100dtxt/glove.6B.100d.txt", help="glove embeding file")
    parser.add_argument('--train_or_restore', type=str, default='Train', help="Train or Restore. (default: Train)")

    # hyper-para for training
    parser.add_argument('--BiLSTM', type=bool, default=False, help="True for wipo/BGC; False for Enron/Reuters.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning Rate.")
    parser.add_argument('--batch_size', type=int, default=30, help="Batch Size (default: 256)")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of training epochs (default: 100)")
    parser.add_argument('--pad_seq_len', type=int, default=300, help="Recommended padding Sequence length of data (depends on the data)")
    parser.add_argument('--embedding_dim', type=int, default=300,help="Dimensionality of character embedding (default: 128)")
    parser.add_argument('--lstm_hidden_size', type=int, default=256,
                        help="Hidden size for bi-lstm layer(default: 256)")
    parser.add_argument('--attention_unit_size', type=int, default=200,
                        help="Attention unit size(default: 200)")
    parser.add_argument('--fc_hidden_size', type=int, default=512,
                        help="Hidden size for fully connected layer (default: 512)")
    parser.add_argument('--dropout', type=float, default=0.5, help= "Dropout keep probability (default: 0.5)")
    parser.add_argument('--l2_reg_lambda', type=float, default= 0.0, help="L2 regularization lambda (default: 0.0)")
    parser.add_argument('--beta', type=float, default=0.5, help="Weight of global scores in scores cal")
    parser.add_argument('--norm_ratio', type=float, default=2, help="The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
    parser.add_argument('--decay_steps', type=int, default=5000,
                        help="The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
    parser.add_argument('--decay_rate', type=float, default=0.95, help="Rate of decay for learning rate. (default: 0.95)")
    parser.add_argument('--checkpoint_every', type=int, default=100, help="Save model after this many steps (default: 100)")
    parser.add_argument('--num_checkpoints', type=int, default=5, help="Number of checkpoints to store (default: 5)")

    # hyper-para for prediction
    parser.add_argument('--evaluate_every', type=int, default=100, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument('--top_num', type=int, default=5, help="Number of top K prediction classes (default: 5)")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for prediction classes (default: 0.5)")


     #日记文件名
    parser.add_argument('--log_file',type=str,default='test.log',help="log result file")
    parser.set_defaults(directed=False)

    return parser.parse_args()
if __name__=='main'():
    args=parse_args()
    train_iter, dev_iter, test_iter = mrs_two(args.datafile_path, args.name_trainfile, args.name_devfile, args.name_testfile, args.char_data, args.text_field,
                                                      args.label_field, repeat=False, shuffle=args.epochs_shuffle, sort=False)
    
    logging.basicConfig=(
        filename=args.log_file,
        endcoding='utf-8',
        level=logging.INFO,
        filemode='w'
    )
    train()