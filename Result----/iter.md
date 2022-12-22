Iter:    200,  Train Loss:  0.32,  Train Acc: 91.41%,  Val Loss:  0.35,  Val Acc: 90.07%,  Time: 1:27:04 *
Iter:    300,  Train Loss:  0.27,  Train Acc: 89.84%,  Val Loss:  0.31,  Val Acc: 90.82%,  Time: 2:06:00 *
Iter:    400,  Train Loss:   0.4,  Train Acc: 87.50%,  Val Loss:  0.28,  Val Acc: 91.71%,  Time: 2:49:03 *


/**
简单来说loss是给深度学习看的，用来优化参数，实现梯度下降。acc是给人看的，用来衡量网络的指标，其实召回率，f1，ap等都是给人看的。
一般来说loss越小，表示网络优化程度高，acc就会越多。但有时也会acc略有下降，总体的趋势是loss下降，acc上升

作者：一枚宅小宋
链接：https://www.zhihu.com/question/264892967/answer/833800306
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

作者：吴昊
链接：https://www.zhihu.com/question/435099359/answer/2457089939
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

其实这个时候要干的事儿很简单，把下面的这种图给画出来就行（图画得丑了点，接受批评）：
<img src="https://picx.zhimg.com/v2-34b42284e073a835f2a18525c105ae2d_b.jpg" data-rawwidth="413" data-rawheight="303" data-size="normal" data-default-watermark-src="https://pic3.zhimg.com/v2-44ab5c200bd1b17ae0caa82bc7086012_b.jpg" class="content_image" width="413"/>图1. Model with high variance<img src="https://pica.zhimg.com/v2-e033b10d5cb3cd33b1735ff1ae9b9290_b.jpg" data-rawwidth="432" data-rawheight="323" data-size="normal" data-default-watermark-src="https://pic3.zhimg.com/v2-a661d11c4ad7ba9dde002ff2b8038210_b.jpg" class="origin_image zh-lightbox-thumb" width="432" data-original="https://pica.zhimg.com/v2-e033b10d5cb3cd33b1735ff1ae9b9290_r.jpg"/>图2. Model with high bias<img src="https://picx.zhimg.com/v2-8ba147a93229b7c51492c7398e54a40b_b.jpg" data-rawwidth="441" data-rawheight="339" data-size="normal" data-default-watermark-src="https://pic1.zhimg.com/v2-cab45b99ea7c90cfab2082e364b3f2b2_b.jpg" class="origin_image zh-lightbox-thumb" width="441" data-original="https://picx.zhimg.com/v2-8ba147a93229b7c51492c7398e54a40b_r.jpg"/>图3. Model with both high variance and high bias我们姑且把上面这种图叫做loss-size图，这里解释一下上面的这种图的意思，纵轴是代表loss，而横轴指的是训练集的大小；要把这张图画出来，需要咱们把训练集划分成很多等分之后，不断扩充训练集的大小来训练模型直到模型收敛位置；比如咱们的训练集包含1000张图片，这样我们可以把训练集随机分成10等分，第一次用100张训练，第二次用200张训练，以此类推，直到最后1000张全部用进去。而上图中，蓝线对应的是随着训练集规模的扩大，validation set上loss的减少，这个很好理解；但是红线的就要稍微解释一下了，红线代表的是，随着training set的增加，模型在training set上loss的增加。为啥training set增加之后模型在training set上的loss会增加？咱们想象一下，极限情况如果training set只有几张图片，此时模型的信息容量实际上是足够把这几张图片的信息全部记住的，但是随着训练集的规模的扩大，模型不可能把这些信息全部记忆下来，那必然会导致一些loss的产生，且这个loss与训练集的规模成正比。好了，因为咱们现在是模型在validation集上的表现不佳，那做出来的loss-size图必然是上面三种情况当中的一种，现在咱们来分别讨论每种图的含义，以及对应的优化策略：2.1 High Variance如果我们做出来的图如上面的图1所示，则说明我们的误差主要来源于Variance误差，所谓的Variance误差指的是模型在验证集上的误差与在训练集上的误差之间的差值。如果遇到这种情况，说明咱们的模型的信息容量是足够应付当前的数据的，但是模型的泛化能力就很差了，说白了就是过拟合了。若模型做出来的图是这种情况，换一个更大的模型是没用的，甚至会起反效果，因为模型大概率已经过拟合了；而对模型做regularization，或者使用drop out这种trick，或者直接扩大训练集的规模都是比较好的优化策略。2.2 High Bias如果做出来的图是图2这种情况，则说明我们的误差主要来源是Bias误差，所谓Bias误差指的是模型在训练集上的误差与理论最小误差之间的差值（这里我们假设理论最小误差为0）.对于这种情况，说明咱们的模型此时是处于一种欠拟合状态的，也就是说模型的信息容量不足。此时盲目增加训练集的规模，或者说做数据增广是没用的，因为多的信息模型也记不住。而增大模型的规模（例如增加更多的网络层数），甚至直接换一个更大的模型则是有效的优化方法。2.3 Both如果是图3的那种情况，那么你就倒霉了，这种情况说明你的模型又记不住信息，泛化能力又差，在这种情况下，上面提到的优化方法都可以用，但还是推荐优先优化模型在训练集上的表现，也就是把模型变大或者换大模型，因为模型的信息容量是基础，模型在生产环境当中的表现不可能优于模型在训练集上的表现。