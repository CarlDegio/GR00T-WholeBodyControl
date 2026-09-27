T2 CHEST 人脸逐帧修正

全局 0 基帧号；1920×1080，全屏整数 exclusive xyxy。仅替换 chest_gray_1/2/3/4 与 chest_white。

已检查全部 319 个实际输出帧（qc_00.jpg 至 qc_15.jpg，非抽样），采用邻近真实人脸模板匹配并逐段人工修正左右边缘半脸。人物完全出画时留空。所有框限制在 CHEST viewport 1488,343,1836,565 内。

数据：masks.json；逐帧详情：details.json；范围与格式校验：verification.json；可复现脚本：refine.py。
