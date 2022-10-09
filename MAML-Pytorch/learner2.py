import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np



class Learner(nn.Module):
    """

    """

    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__() # 这里是在调用父类构造函数，对父类进行初始化吗？


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]，ch_out处于最高维度，ch_out代表的是卷积核的个数，也代表了输出特征图的channel数
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                u,s,v = torch.svd(w)
                
                self.vars.append(w)
                # [ch_out]，这是什么东西？？答：每个卷积核都有一个偏置值b，这是偏置值的意思。
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                
                self.vars.append(nn.Parameter(u))
                self.vars.append(nn.Parameter(s))
                self.vars.append(nn.Parameter(v))

            elif name == 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4])) 
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError




    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True,finetunning=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]，这里应该写错了，应该是x: [batch, channel, width, height]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name == 'conv2d':
                # w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!

                if finetunning:
                    w, b = vars[idx], vars[idx + 1]
                    x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    vars[idx+2], vars[idx + 3],vars[idx+4] = torch.svd(w)
                    idx += 5 # 取出了权重w和偏置b，所以需要加+2
                else:
                    u,s,v = vars[idx+2], vars[idx + 3],vars[idx+4]
                    u = u.detach()
                    v = v.detach()
                    x = F.conv2d(x, v , stride=param[4], padding=param[5])
                    x = F.conv2d(x, s.diag_embed().transpose(0,1), stride=param[4], padding=param[5])
                    x = F.conv2d(x, u, stride=param[4], padding=param[5])
                    idx += 5 # 取出了权重w和偏置b，所以需要加+2
                # u.requires_grad = True
                # s.requires_grad = True
                # v.requires_grad = True
                                
                # x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                

                # print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5]) # 这里执行了逆卷积：https://blog.csdn.net/qq_27261889/article/details/86304061
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return x


    def updateUSV(self):
        
        idx = 0

        for name, param in self.config:
            if name == 'conv2d':
                self.vars[idx+2], self.vars[idx + 3],self.vars[idx+4] = torch.svd(self.vars[idx])
                idx += 5
            elif name == 'convt2d':
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                idx += 2
            else:
                raise NotImplementedError


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        # 将输入的var置空，或将本类的var置空
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars