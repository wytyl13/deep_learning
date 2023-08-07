/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-08-07 10:58:06
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-08-07 10:58:06
 * @Description: 
 * 
 * 
 * 
 * 解决github无法连接错误 OpenSSL SSL_connect: Connection was reset in connection to github.com:443
 * this error generally happend when you used ssl to connect github. and you 
 * used the proxy. so you should define the proxy for git config.
 * git config --global http.proxy 127.0.0.1:7890
 * git config --global https.proxy 127.0.0.1:7890
 * 7890 is the port of your proxy.
***********************************************************************/