# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam


# 权重初始化，默认xavier
# def init_network(model, method='xavier', exclude='embedding', seed=123):
#     for name, w in model.named_parameters():
#         if exclude not in name:
#             if len(w.size()) < 2:
#                 continue
#             if 'weight' in name:
#                 if method == 'xavier':
#                     nn.init.xavier_normal_(w)
#                 elif method == 'kaiming':
#                     nn.init.kaiming_normal_(w)
#                 else:
#                     nn.init.normal_(w)
#             elif 'bias' in name:
#                 nn.init.constant_(w, 0)
#             else:
#                 pass


def train(config, model, train_iter, dev_iter, test_iter):
    # 开始时间
    start_time = time.time()
    # 开始训练
    model.train()
    # 参数优化？
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    # 记录进行到多少batch
    total_batch = 0
    dev_best_loss = float('inf')
    # 记录上次验证集loss下降的batch数
    last_improve = 0
    # 记录是否很久没有效果提升
    flag = False
    # 开始训练
    model.train()
    flag_print_trains_shape=True
    # 3 rounds
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            # print(str(i)+"-------"+str(len(train_iter))+"------" +str(trains)+"!"+str(labels))
            print(str(i) + "-------" + str(len(train_iter)) )
            if flag_print_trains_shape:
                print("hhhhh")
                print(trains)
                # hhhhh-0torch.Size([64, 256])
                print("hhhhh-0"+str(trains[0].shape))
                print(trains[0])
                # hhhhh-1torch.Size([64])
                print("hhhhh-1"+str(trains[1].shape))
                print(trains[1])
                # hhhhh-2torch.Size([64, 256])
                print("hhhhh-2"+str(trains[2].shape))
                print(trains[2])
                # labelstorch.Size([64])
                print("hhhhh-labels"+str(labels.shape))
                print(labels)
                flag_print_trains_shape=False
            # 训练并且得到训练输出
            outputs = model(trains)
            #
            model.zero_grad()
            # 计算loss
            loss = F.cross_entropy(outputs, labels)
            # 反向传播
            loss.backward()
            # 优化器
            optimizer.step()
            # todo 100轮才打印一回，怪不得看不到
            if total_batch % 1 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                # 真实标签结果
                true = labels.data.cpu()
                # 模型预测输出标签
                predic = torch.max(outputs.data, 1)[1].cpu()
                # 训练准确率
                train_acc = metrics.accuracy_score(true, predic)
                # 验证集 evaluate
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            # # 若超过1000batch效果还没提升，则提前结束训练
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    # 完成训练之后进行测试
    test(config, model, test_iter)

# 精度(precision) = 正确预测的个数(TP)/被预测正确的个数(TP+FP)
# 召回率(recall)=正确预测的个数(TP)/预测个数(TP+FN)
# F1 = 2*精度*召回率/(精度+召回率)
# support: 当前行的类别在测试数据中的样本总量
# 同时还会给出总体的微平均值，宏平均值和加权平均值。
# accuracy：计算正确率 (TP+TN) / (TP＋FP＋FN＋TN)
# macro avg：各类的precision，recall，f1加和求平均
# weighted avg :对每一类别的f1_score进行加权平均，权重为各类别数在y_true中所占比例
# 链接：https://www.jianshu.com/p/4b4530f7ea3f
# http://www.manongjc.com/detail/31-vnhrdhlxiqovkcj.html
#https://blog.csdn.net/kancy110/article/details/74937469
#
def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    # evaluate  return acc, loss_total / len(data_iter), report, confusion
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            # 模型输出
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            # 标签
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    print(labels_all)
    print(predict_all)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
