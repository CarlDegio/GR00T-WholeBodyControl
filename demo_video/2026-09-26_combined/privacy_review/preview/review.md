# 四宫格与OVMM隐私审查

5.0–45.0秒每0.5秒抽取81帧，每帧四分区分别放大审查，共324分区；OVMM45.3–54.8秒每0.5秒审查20帧，并查看55.3秒切点。T3初段和T4短露脸用0.1秒加密，T2人脸出现/消失边界0.1/0.25秒加密。截图、拼图全部保留。

1920x1080全屏；bbox=[x1,y1,x2,y2]，仅示例帧位置，不是整段固定遮罩。

|完整视频时间(s)|位置|类型|处理|理由|示例截图|
|---|---|---|---|---|---|
|5.000–45.300|右上 T2|确认纸箱二维码（前景双码）|必须打码|前景纸箱正面的两个二维码反复清晰出现，搬动、转箱及末帧停留均继续曝光；建议整段跟踪两码，转向时覆盖可见码面。|[P1](/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_combined/privacy_review/preview/evidence_P1_5.000.jpg)|
|5.000–15.770|右上 T2 背景红色椅背左侧|小尺寸、部分遮挡人脸|保守建议|可确认真人头部及一小部分侧脸/上脸，脸很小且被椅背与桌面物体遮挡，清晰程度有限。|[P2](/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_combined/privacy_review/preview/evidence_P2_5.000.jpg)|
|23.533–31.800|右上 T2 背景红色椅背上方|确认真人侧脸|必须打码|坐在桌边的人员侧脸露出，放大后可见脸部轮廓与五官，不是机器人；镜头下移后逐渐被上边缘裁掉。|[P3](/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_combined/privacy_review/preview/evidence_P3_24.000.jpg)|
|31.800–35.600|右上 T2 顶缘红色椅背上方|裁切的下半脸/脸部边缘|保守建议|P3同一人员的头部逐步出画，部分时段仅残留脸颊/下脸，建议延续遮挡到35.6秒避免边缘闪漏。|[P4](/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_combined/privacy_review/preview/evidence_P4_33.500.jpg)|
|5.000–12.467|左下 T3 上边缘、机器人左侧|确认真人侧脸及裁切面部|必须打码|5.0–5.2、6.4–6.8秒侧脸明显；其余时间多数上脸出画但下半脸仍可见，建议该镜头持续跟踪至12.467秒切镜。|[P5](/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_combined/privacy_review/preview/evidence_P5_6.500.jpg)|
|33.800–34.300|右下 T4 最左边缘|短暂真人正/侧脸|必须打码|33.9–34.1秒明确露脸，0.1秒细查显示出入画发生在33.8、34.2秒附近；建议两端留余量。|[P6](/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_combined/privacy_review/preview/evidence_P6_34.000.jpg)|
|43.750–45.300|右下 T4 左上角边缘|局部下颌/颈部边缘|保守建议|黄衣人员进入左缘，完整脸在画面外，43.9–44.0秒接近顶部可能残留下颌；后续含末帧停留多为颈部/身体。不应称为清晰整脸。|[P7](/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_combined/privacy_review/preview/evidence_P7_44.000.jpg)|
|5.000–23.533|右下 T4 左侧背景推车纸箱|确认纸箱二维码（小双码）|必须打码|与T2同类箱体正面双码在背景持续可见，虽然尺寸小，仍具有码块纹理。|[P8](/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_combined/privacy_review/preview/evidence_P8_5.000.jpg)|
|5.000–23.533|右下 T4 最右侧背景纸箱|确认纸箱二维码（部分切边）|必须打码；早段部分被遮/切边|16.5–23.533秒右侧纸箱双码明确露出；前段也有箱面边缘，建议整段覆盖实际可见码块以防间歇漏码。|[P9](/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_combined/privacy_review/preview/evidence_P9_19.000.jpg)|
|23.533–45.300|右下 T4 后方中左推车纸箱|确认纸箱二维码（小双码）|必须打码|切入远景后纸箱移到中左背景，双码持续可见且有机器人腿部短时遮挡，末帧停留也应保留遮挡。|[P10](/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_combined/privacy_review/preview/evidence_P10_25.000.jpg)|
|5.000–45.300|右上 T2 后方堆叠纸箱|远景方形标签/疑似二维码|保守建议|背景其他箱体上也有小型方形印刷块；仅凭当前成片像素无法对每一块确认二维码，建议遮挡可疑方码，勿把蓝色圆形图案全部当二维码。|[P11](/home/user/Project/GR00T-WholeBodyControl/demo_video/2026-09-26_combined/privacy_review/preview/evidence_P11_40.000.jpg)|

## 无需列入项
- T1所有抽样未见真实人脸或纸箱二维码。机器人蓝灯头部、躯干不计为人脸。
- T3纸箱可见短侧面主要为卡通/图标，未确认到二维码；未据此误报箱码。
- OVMM45.3–55.3秒为仿真对比画面，审查样本未见真实人脸或纸箱二维码。场景挂画未发现可辨认真人脸，机器人模型不是隐私人脸。

## 不确定项
- 这是密集抽帧人工审查，非每帧证明。时间为建议包络，短暂遮挡或码面背向时不必打空处；实际打码应逐帧跟踪。
- T2背景远景方块未扫码，未对内容或可解码性作推断。
- T2初段小侧脸、后段裁切下脸及T4末段下颌边缘均已单列保守项，不应当成清晰整脸。