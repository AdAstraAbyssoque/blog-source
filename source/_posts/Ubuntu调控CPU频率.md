---
title: Ubuntu调控CPU频率
date: 2024-10-08 23:50:51
tags: 技术
cover: https://ooo.0x0.ooo/2024/10/09/ODBXwI.webp
---

## 问题描述

跑神经网络的时候可能会遇到高并发的情况，这种情况下会带来很高的 CPU 负载，系统层面需要调优 CPU 的性能。

查看各 CPU 核心的工作模式可以通过下面这条命令：

```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

终端打印以下内容：

```txt
powersave
powersave
powersave
powersave
powersave
powersave
powersave
powersave
powersave
powersave
powersave
powersave
```

说明 CPU 正在养生...

## 解决方案

安装 indicator-cpufreq，在图形化顶栏即可调优。

```bash
sudo apt-get install indicator-cpufreq
```

或使用`cpufrequtils`:

```bash
sudo apt-get install cpufrequtils
```

运行下面的修改

```bash
cpu_mode=performance
# Get the number of CPU cores
cpu_cores=$(nproc)

# Set each CPU core to performance mode
for ((cpu=0; cpu<cpu_cores; cpu++)); do
  sudo cpufreq-set -c $cpu -g $cpu_mode
done
```

同时可选择设置开机启动等附加内容。

## Reference

1. [Blog](https://blog.csdn.net/xuzhengzhe/article/details/137066275#:~:text=%E7%84%B6%E8%80%8C%EF%BC%8C%E9%BB%98%E8%AE%A4%E6%83%85%E5%86%B5%E4%B8%8B%EF%BC%8CU)
