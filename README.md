# Video Object Segmentation Survey

## Table of content

- [Dataset](## Dataset)
- [Papers](## Papers)
- [Codebase](## Codebase)

## Dataset

**Datasets collected for video object segmentation problem**

|      name      |                           website                            |          description           |
| :------------: | :----------------------------------------------------------: | :----------------------------: |
|     DAVIS      | [https://davischallenge.org/index.html](https://davischallenge.org/index.html) |   Densely annotated datasets   |
| Youtube-Object | [https://data.vision.ee.ethz.ch/cvl/youtube-objects/](https://data.vision.ee.ethz.ch/cvl/youtube-objects/) |   Densely annotated datasets   |
|      FBMS      | [https://lmb.informatik.uni-freiburg.de/resources/datasets/](https://lmb.informatik.uni-freiburg.de/resources/datasets/) |   Densely annotated datasets   |
|  Youtube-VOS   |     [https://youtube-vos.org](https://youtube-vos.org/)      | Large scale, sparse annotation |
|    SegTrack    | [https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html](https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html) |  Full pixel-level annotation   |
|    TAO-VOS     | [www.vision. rwth-aachen.de/page/taovos](www.vision. rwth-aachen.de/page/taovos) |   pseudo labels are included   |

**Additional datasets used for data augmentation or pretraining**

Supervised pretraining:  [PASCAL-VOC](http://host.robots.ox.ac.uk/pascal/VOC/), [COCO](https://cocodataset.org/), [MSRA10K](https://mmcheng.net/msra10k/), [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [HKU-IS](https://sites.google.com/site/ligb86/mdfsaliency/)

Large scale self-supervised pretraining: [Kinetics](https://deepmind.com/research/open-source/kinetics), [VLOG](https://web.eecs.umich.edu/~fouhey/2017/VLOG/index.html), [OxUva](https://oxuva.github.io/long-term-tracking-benchmark/)

## Papers

**Semi-supervised VOS**

Online Adaptation of Convolutional Neural Networks for Video Object Segmentation. BMVC 2017 [[Paper](http://www.bmva.org/bmvc/2017/papers/paper116/paper116.pdf)] [[Project Page](https://www.vision.rwth-aachen.de/page/OnAVOS)]

One-Shot Video Object Segmentation. CVPR 2017 [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Caelles_One-Shot_Video_Object_CVPR_2017_paper.pdf)] [[Project Page](https://cvlsegmentation.github.io/osvos/)] [[CAFFE Code](https://github.com/kmaninis/OSVOS-caffe)] [[TensorFlow Code](https://github.com/scaelles/OSVOS-TensorFlow)] [[Pytorch Code](https://github.com/kmaninis/OSVOS-PyTorch)]

Learning Video Object Segmentation from Static Images. CVPR 2017 [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Perazzi_Learning_Video_Object_CVPR_2017_paper.pdf)] [[Pytorch Code](https://github.com/omkar13/MaskTrack)]

FusionSeg: Learning to combine motion and appearance for fully automatic segmentation of generic objects in videos. CVPR 2017 [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Jain_FusionSeg_Learning_to_CVPR_2017_paper.pdf)] [[Project Page](http://vision.cs.utexas.edu/projects/fusionseg/)] [[CAFFE Code](https://github.com/suyogduttjain/fusionseg)]

Lucid Data Dreaming for Video Object Segmentation. CVPR 2017 Workshop [[Paper](https://arxiv.org/abs/1703.09554)] [[Code](https://github.com/ankhoreva/LucidDataDreaming)]

MaskRNN: Instance Level Video ObjectSegmentation. NeurIPS 2017 [[Paper](https://papers.nips.cc/paper/2017/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)] 

Efficient Video Object Segmentation via Network Modulation. CVPR 2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Efficient_Video_Object_CVPR_2018_paper.pdf)] [[TensorFlow Code](https://github.com/linjieyangsc/video_seg)]

Fast Video Object Segmentation by Reference-Guided Mask Propagation. CVPR 2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Oh_Fast_Video_Object_CVPR_2018_paper.pdf)] [[PyTorch Code](https://github.com/seoungwugoh/RGMP)]

Fast and Accurate Online Video Object Segmentation via Tracking Parts. CVPR 2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Cheng_Fast_and_Accurate_CVPR_2018_paper.pdf)] [[CAFFE Code](https://github.com/JingchunCheng/FAVOS)]

VideoMatch: Matching based Video Object Segmentation. ECCV 2018 [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuan-Ting_Hu_VideoMatch_Matching_based_ECCV_2018_paper.pdf)] [[Pytorch Code](https://github.com/stashvala/Pytorch-VideoMatch)]

Video Object Segmentation with Joint Re-identification and Attention-Aware Mask Propagation. ECCV 2018 [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaoxiao_Li_Video_Object_Segmentation_ECCV_2018_paper.pdf)] 

PReMVOS: Proposal-generation, Refinement and Merging for Video Object Segmentation. ACCV 2018 [[Paper](https://arxiv.org/abs/1807.09190)] [[Code](https://github.com/JonathonLuiten/PReMVOS)]

A Generative Appearance Model for End-to-end Video Object Segmentation. CVPR 2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Johnander_A_Generative_Appearance_Model_for_End-To-End_Video_Object_Segmentation_CVPR_2019_paper.pdf)] [[Pytorch Code](https://github.com/joakimjohnander/agame-vos)]

MHP-VOS: Multiple Hypotheses Propagation for Video Object Segmentation. CVPR 2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_MHP-VOS_Multiple_Hypotheses_Propagation_for_Video_Object_Segmentation_CVPR_2019_paper.pdf)] [[Code](https://github.com/shuangjiexu/MHP-VOS)]

Spatiotemporal CNN for Video Object Segmentation. CVPR 2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Spatiotemporal_CNN_for_Video_Object_Segmentation_CVPR_2019_paper.pdf)] [[Pytorch Code](https://github.com/longyin880815/STCNN)]

Fast Online Object Tracking and Segmentation: A Unifying Approach. CVPR 2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Fast_Online_Object_Tracking_and_Segmentation_A_Unifying_Approach_CVPR_2019_paper.html)] [[Pytorch Code](https://github.com/foolwood/SiamMask)]

See More, Know More: Unsupervised Video Object Segmentation With Co-Attention Siamese Networks. CVPR 2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Lu_See_More_Know_More_Unsupervised_Video_Object_Segmentation_With_Co-Attention_CVPR_2019_paper.html)] [[Pytorch Code](https://github.com/carrierlxk/COSNet)]

RVOS: End-To-End Recurrent Network for Video Object Segmentation. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Ventura_RVOS_End-To-End_Recurrent_Network_for_Video_Object_Segmentation_CVPR_2019_paper.html)] [[Project Page](https://imatge-upc.github.io/rvos/)] [[Pytorch Code](https://github.com/imatge-upc/rvos)]

BubbleNets: Learning to Select the Guidance Frame in Video Object Segmentation by Deep Sorting Frames. CVPR 2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Griffin_BubbleNets_Learning_to_Select_the_Guidance_Frame_in_Video_Object_CVPR_2019_paper.pdf)] [[Code](https://github.com/griffbr/BubbleNets)]

FEELVOS: Fast End-to-End Embedding Learning for Video Object Segmentation. CVPR 2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Voigtlaender_FEELVOS_Fast_End-To-End_Embedding_Learning_for_Video_Object_Segmentation_CVPR_2019_paper.pdf)] [[TensorFlow Code](https://github.com/tensorflow/ models/tree/master/research/feelvos)]

DMM-Net: Differentiable Mask-Matching Network for Video Object Segmentation. ICCV 2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zeng_DMM-Net_Differentiable_Mask-Matching_Network_for_Video_Object_Segmentation_ICCV_2019_paper.pdf)] [[Pytorch Code](https://github.com/ZENGXH/DMM_Net)] 

AGSS-VOS: Attention Guided Single-Shot Video Object Segmentation. ICCV 2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_AGSS-VOS_Attention_Guided_Single-Shot_Video_Object_Segmentation_ICCV_2019_paper.html)] [[Pytorch Code](https://github.com/Jia-Research-Lab/AGSS-VOS)]

RANet: Ranking Attention Network for Fast Video Object Segmentation. ICCV 2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_RANet_Ranking_Attention_Network_for_Fast_Video_Object_Segmentation_ICCV_2019_paper.pdf)] [[Pytorch Code](https://github.com/Storife/RANet)]

CapsuleVOS: Semi-Supervised Video Object Segmentation Using Capsule Routing. ICCV 2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Duarte_CapsuleVOS_Semi-Supervised_Video_Object_Segmentation_Using_Capsule_Routing_ICCV_2019_paper.pdf)] [[Pytorch Code](https://github.com/KevinDuarte/CapsuleVOS)]

Video Object Segmentation using Space-Time Memory Networks. ICCV 2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.pdf)] [[Pytorch Code](https://github.com/seoungwugoh/STM)]

State-Aware Tracker for Real-Time Video Object Segmentation. CVPR 2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_State-Aware_Tracker_for_Real-Time_Video_Object_Segmentation_CVPR_2020_paper.pdf)] [[Pytorch Code](https://github.com/MegviiDetection/video_analyst)]

A Transductive Approach for Video Object Segmentation. CVPR 2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_A_Transductive_Approach_for_Video_Object_Segmentation_CVPR_2020_paper.pdf)] [[Pytorch Code](https://github.com/ microsoft/transductive-vos.pytorch)]

Learning Fast and Robust Target Models for Video Object Segmentation. CVPR 2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Robinson_Learning_Fast_and_Robust_Target_Models_for_Video_Object_Segmentation_CVPR_2020_paper.pdf)] [[Pytorch Code](https://github.com/andr345/frtm-vos)]

Learning What to Learn for Video Object Segmentation. ECCV 2020 [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4440_ECCV_2020_paper.php)] [[Pytorch Code](https://github.com/visionml/pytracking)]

Video Object Segmentation with Episodic Graph Memory Networks. ECCV 2020 [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480664.pdf)] [[Pytorch Code](https://github.com/carrierlxk/GraphMemVOS)]

Collaborative Video Object Segmentation by Foreground-Background Integration. ECCV 2020 [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500324.pdf)] [[Pytorch Code](https://github.com/z-x-yang/CFBI)]

Kernelized Memory Network for Video Object Segmentation.  ECCV 2020 [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4152_ECCV_2020_paper.php)] 

Delving into the Cyclic Mechanism in Semi-supervised Video Object Segmentation. NeurIPS 2020 [[Paper](https://papers.nips.cc/paper/2020/file/0d5bd023a3ee11c7abca5b42a93c4866-Paper.pdf)] [[Pytorch Code](https://github.com/lyxok1/STM-Training)]

Make One-Shot Video Object Segmentation Efficient Again. NeurIPS 2020 [[Paper](https://papers.nips.cc/paper/2020/file/781397bc0630d47ab531ea850bddcf63-Paper.pdf)] [[Pytorch Code](https://github.com/dvl-tum/e-osvos)]

Video Object Segmentation with Adaptive Feature Bank and Uncertain-Region Refinement NeurIPS 2020 [[Paper](https://papers.nips.cc/paper/2020/file/234833147b97bb6aed53a8f4f1c7a7d8-Paper.pdf)] [[Pytorch Code](https://github.com/xmlyqing00/AFB-URR)]



**Unsupervised VOS**

Video Segmentation via Object Flow. CVPR 2016 [[Paper](https://faculty.ucmerced.edu/mhyang/papers/cvpr16_object_flow.pdf)] [[CAFFE Code](https://github.com/wasidennis/ObjectFlow)]

Learning Unsupervised Video Object Segmentation through Visual Attention. CVPR 2019 [[Paper](https://www.researchgate.net/publication/332751903_Learning_Unsupervised_Video_Object_Segmentation_through_Visual_Attention)] [[Journal](https://www.researchgate.net/publication/338528322_Paying_Attention_to_Video_Object_Pattern_Understanding)] [[CAFFE Code](https://github.com/wenguanwang/AGS)]

Zero-shot Video Object Segmentation via Attentive Graph Neural Networks.  ICCV 2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Zero-Shot_Video_Object_Segmentation_via_Attentive_Graph_Neural_Networks_ICCV_2019_paper.pdf)] [[Pytorch Code](Zero-shot Video Object Segmentation via Attentive Graph Neural Networks.  ICCV 2019)]

Anchor Diffusion for Unsupervised Video Object Segmentation. ICCV 2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_Anchor_Diffusion_for_Unsupervised_Video_Object_Segmentation_ICCV_2019_paper.html)] [[Pytorch Code](https://github.com/yz93/anchor-diff-VOS)]

UnOVOST: Unsupervised Offline Video Object Segmentation and Tracking. WACV 2020 [[Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Luiten_UnOVOST_Unsupervised_Offline_Video_Object_Segmentation_and_Tracking_WACV_2020_paper.pdf)] [[Pytorch Code](https://github.com/idilesenzulfikar/UNOVOST)]

Unsupervised Video Object Segmentation with Joint Hotspot Tracking. ECCV 2020 [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590477.pdf)] [[Code](https://github.com/luzhangada/code-for-WCS-Net)]



**Interactive VOS**

Fast User-Guided Video Object Segmentation by Interaction-And-Propagation Networks. CVPR 2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Oh_Fast_User-Guided_Video_Object_Segmentation_by_Interaction-And-Propagation_Networks_CVPR_2019_paper.html)] [[Pytorch Code](https://github.com/seoungwugoh/ivs-demo)]

Interactive Video Object Segmentation Using Global and Local Transfer Modules. ECCV 2020 [[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620290.pdf)] [[Pytorch Code](https://github.com/yuk6heo/IVOS-ATNet)]



**Representation Learning for correspondence flow**

Tracking Emerges by Colorizing Videos. ECCV 2018 [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Carl_Vondrick_Self-supervised_Tracking_by_ECCV_2018_paper.pdf)]

Self-supervised Learning for Video Correspondence Flow. BMVC 2019 [[Paper](https://bmvc2019.org/wp-content/uploads/papers/0599-paper.pdf)] [[Pytorch Code](https://bmvc2019.org/wp-content/uploads/papers/0599-paper.pdf)]

Learning Correspondence from the Cycle-consistency of Time. CVPR 2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_Correspondence_From_the_Cycle-Consistency_of_Time_CVPR_2019_paper.pdf)] [[Pytorch Code](https://github.com/xiaolonw/TimeCycle)] 

Learning Video Object Segmentation from Unlabeled Videos. CVPR 2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_Learning_Video_Object_Segmentation_From_Unlabeled_Videos_CVPR_2020_paper.pdf)] [[Pytorch Code](https://github.com/carrierlxk/MuG)]

Space-Time Correspondence as a Contrastive Random Walk. NeurIPS 2020 [[Paper](https://proceedings.neurips.cc/paper/2020/file/e2ef524fbf3d9fe611d5a8e90fefdc9c-Paper.pdf)] [[Project Page](https://ajabri.github.io/videowalk/)] [[Pytorch Code](https://github.com/ajabri/videowalk)]

## Codebase

- [pysot](https://github.com/STVIR/pysot)
- [pytracking](https://github.com/visionml/pytracking)
- [video-analyst](https://github.com/MegviiDetection/video_analyst)

