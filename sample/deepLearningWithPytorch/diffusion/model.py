'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-12 21:50:33
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-12 21:50:33
 * @Description: diffusion is dedicated to generating picture based on the init noise image.
 * it will expand the noise image and generate the picture based on the keyword.
 * but it is better than GAN.
 * GAN有很多劣势：
        首先GAN需要训练两个模型，一个是生成器一个判别器。收敛起来困难
        其次GAN的生成器只需要对抗判别器即可，所以学习的不彻底，容易走捷径。
        最后GAN的多样性差，因为它只需要对抗判别器。
 * 然后GAN的劣势可以使用扩散模型化解。
 * 扩散模型就是对初始噪音图像不断去噪，然后生成我们想要的图像的一个过程，那么在学习如何
 * 去噪之前，我们先要学习噪音如何影响一个图像的。注意前提是这个噪声满足标准的正态分布。

 * DDPM: denoising diffusion probability model.去噪声扩散概率模型
 * 训练的过程包括前向和后向两个部分，使用一张原始图像，使用高斯概率分布的噪声去逐步影响原始图像
 * 最终得到的一个纯高斯噪声图像就是该原始图像对应的标签，注意这个前向过程是线性固定的，所以
 * 不存在任何可学习的参数，后向的过程是存在学习参数的。
 * 我们使用前向过程得到的噪声图像作为原始图像的标签，使用DDPM模型去训练对应的参数。最后得到的模型
 * 就是diffusion模型。使用它可以将任何随机初始化高斯分布的噪声生成我们想要的图像。当然，
 * 生成图像的风格和训练样本的风格是一样的。所以才存在扩散模型风格一说。因为不存在可以涵盖
 * 所有形式的风格的模型。其实这个模型中后向过程是重点。但是我们在开始后向的探讨之前有必要了解
 * 下前向的内容。
 * 前向使一个线性的过程，也即从原始图像到最后生成的纯噪声图像，这中间的每一步都是由前一步的结果
 * 计算得到的，而且是线性的函数。那么问题来了，我们是否有必要像RNN一样对每一步进行计算呢？
 * 没有必要，因为我们前向的过程不需要grad，也不需要知道前向传播中每一步的结果。所以我们可以直接
 * 根据函数计算得到最后一步的结果。
***********************************************************************'''  