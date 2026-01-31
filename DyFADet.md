# DyFADet: Dynamic Feature Aggregation forTemporal Action Detection

Le $\mathrm { Y a n g ^ { 1 * ( \bigtriangledown ) } } \oplus$ , Ziwei Zheng1∗ , Yizeng Han2 , Hao Cheng3,Shiji Song $^ 4 \oplus$ ： , Gao Huang $^ 4 \oplus$ , and Fan Li1

1 Xi’an Jiaotong University{yangle15, lifan}@xjtu.edu.cn, ziwei.zheng@stu.xjtu.edu.cn

2 Alibaba Group yizeng38@gmail.com

3 HKUST(GZ) hcheng046@connect.hkust-gz.edu.cn

4 Tsinghua University{shijis, gaohuang}@tsinghua.edu.cn

Abstract. Recent proposed neural network-based Temporal Action De-tection (TAD) models are inherently limited to extracting the discrimina-tive representations and modeling action instances with various lengthsfrom complex scenes by shared-weights detection heads. Inspired by thesuccesses in dynamic neural networks, in this paper, we build a novel dy-namic feature aggregation (DFA) module that can simultaneously adaptkernel weights and receptive fields at different timestamps. Based onDFA, the proposed dynamic encoder layer aggregates the temporal fea-tures within the action time ranges and guarantees the discriminabilityof the extracted representations. Moreover, using DFA helps to developa Dynamic TAD head (DyHead), which adaptively aggregates the multi-scale features with adjusted parameters and learned receptive fields bet-ter to detect the action instances with diverse ranges from videos. Withthe proposed encoder layer and DyHead, a new dynamic TAD model, Dy-FADet, achieves promising performance on a series of challenging TADbenchmarks, including HACS-Segment, THUMOS14, ActivityNet-1.3,Epic-Kitchen 100, Ego4D-Moment QueriesV1.0, and FineAction. Codeis released to https://github.com/yangle15/DyFADet-pytorch.

Keywords: Temporal action detection $\cdot$ Dynamic network architectures· Video understanding

# 1 Introduction

As a challenging and essential task within the field of video understanding,Temporal Action Detection (TAD) has received widespread attention in recentyears. The target of TAD is to simultaneously recognize action categories andlocalize action temporal boundaries from an untrimmed video. Various methodshave been developed to address this task, which can be mainly divided into twocategories: 1) Two-stage methods (such as [39, 45]), which first learn to generate

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/030a8b653963947163d1b20ec3f8358d9c5215fb8197001e052611cc249009b8.jpg)



Fig. 1: Differences between convolution and DFA. (a) A normal convolution with staticweights and receptive fields. (b) The dynamic formations of DFA at different times-tamps. (c) Implementing DFA to build DyFADet can address the two issues in TAD.


the class-agnostic action proposals and then conduct classification and boundary-refinement in proposal-level. 2) One-stage methods [26, 42, 47, 62] classify eachframe as well as locate temporal boundaries in an end-to-end manner, achievingbetter performance and becoming a popular line of TAD research currently.

However, accurately detecting an action from an untrimmed video remains achallenging task. On the one hand, the spatial redundancy in the adjacent framesalong with the static feature extraction strategy can result in poor discriminabil-ity of learned representations, hindering the detection performance [42, 47]. Onthe other hand, the head inadaptation issue can happen in common TAD de-signs [62], where a static shared-weight head is used to detect action instanceswith diverse temporal lengths, leading to sub-optimal performance. Therefore,it is necessary to find a solution that can simultaneously address these two keyissues in modern TAD models.

In this paper, inspired by the recent success of dynamic neural networks [16,22], we develop a novel Dynamic Feature Aggregation (DFA) module for TAD.As is shown in Fig. 1, the proposed DFA, the kernel shape, and the networkparameters can adapt based on input during generating new features, which sig-nificantly differs from the static feature learning procedure in most existing TADmodels. Such a learning mechanism enables a dynamic feature aggregation pro-cedure, which has the ability to increase the discriminability of the learned rep-resentations, and also adapts the feature learning procedure in detection headsbased on each level of the feature pyramid during detection.

Therefore, based on the DFA, we build a dynamic encoder (DynE) and aDynamic TAD head (DyHead) to address the two issues in modern TAD models.On the one hand, the dynamic feature learning of DynE can facilitate the featuresof target action to be gathered together and increase the differences between thefeatures of the action and boundaries, resolving the first issue in Fig. 1. On theother hand, the DyHead will dynamically adjust detector parameters when it isapplied at different pyramid levels, which corresponds to detecting the actionswith different time ranges. By effectively addressing the two issues in Fig. 1,the proposed DyFADet with DynE and DyHead can achieve accurate detectionresults in TAD tasks.

We evaluate the proposed DyFADet on a series of TAD datasets includingHACS-Segment [64], THUMOS14 [23], ActivityNet-1.3 [5], Epic-Kitchen 100 [13],Ego4D Moment Queries V1.0 [15] and FineAction [33]. The experimental resultsshow the effectiveness of the proposed DyFADet in TAD tasks.

# 2 Related Work

Temporal action detection is a challenging video understanding task, whichinvolves localizing and classifying the actions in an untrimmed video. Conven-tional two-stage approaches $\lvert 8 , 2 6 , 2 8 , 2 9 , 3 9 , 4 5 , 5 7 , 6 0 , 6 1 , 6 8 \rvert$ usually consist oftwo steps: proposal generation and classification. Nevertheless, these may suf-fer from high complexity and end-to-end training could be infeasible. A recenttrend is designing one-stage frameworks and training the model in an end-to-endfashion. Some works [32, 43, 46] propose to detect actions with the DETR-likedecoders [6], and another line of work [10, 42, 62] builds a multi-scale featurepyramid followed by a detection head. In this work, we mainly follow the main-stream encoder-pyramid-head framework. Concretely, we build our model basedon the frameworks in the highly competitive methods [42, 62], and boost thefinal detection performance by introducing the dynamic mechanism to simul-taneously solve the less-discriminant feature and the head inadaption issues inTAD models.

Dynamic neural networks have attracted great research interests in recentyears due to their favorable efficiency and representation power [16]. Unlike con-ventional models that process different inputs with the same computation, dy-namic networks can adapt their architectures and parameters conditioned oninput samples [18, 21, 59] or different spatial [17, 19, 51] / temporal [38, 55] posi-tions.

Data-dependent parameters have shown effectiveness in increasing the rep-resentation power with minor computational overhead [9, 12, 36, 58]. Existingapproaches can generally be divided into two groups: one adopts mechanismsto dynamically re-weight parameter values [9, 36, 58], including modern self-attention operators, which can be interpreted as (depth-wise) dynamic convolu-tions [67]. While the static temporal reception fields of these methods can leadto the less-discriminant feature issue as stated in [42,47]. The other develops de-formable convolution to achieve dynamic reception fields [12], which have beeneffectively utilized in different video understanding tasks [24, 25, 37]. However,the kernel weights of these deformable convolutions are static. Compared to theprevious methods, our DFA simultaneously adapts the convolution weights andtemporal reception fields in a data-dependent manner, leading to more effectiveand flexible modeling of temporal information in the TAD task.

# 3 Method

In this section, we first introduce the proposed Dynamic feature aggregation(DFA) and then develop the dynamic feature learning based TAD model, Dy-FADet for TAD tasks.

# 3.1 Dynamic feature aggregation

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/dff4cd5d6e29e9bc8c3b9e846b8a779ed40d32480375f391bacd9291f0f71ce4.jpg)



(b) A DFA module in TAD.



Fig. 2: An illustration of the proposed DFA module. (a) A normal convolution with thekernel size of 3 (Conv 3) and its corresponding formation realized by a shifting moduleand a point-wise convolution. (b) A DFA. The shifted representations multiplied withthe weighted mask will be sent to the point-wise convolution. The DFA module isequivalent to a convolution with adaptive kernel weights and receptive fields.


DFA based on feature shifting. The Dynamic feature aggregation (DFA)needs to effectively adapt the kernel weights and the receptive fields (includingthe number of sampling positions and the shape of kernels) based on the inputsto improve the feature extraction ability, which is difficult to simultaneouslyachieve. However, such a procedure can be realized if we separate a normal con-volution into a feature-shifting module and a point-wise convolution [53] (shownin Fig. 2): A convolution with the kernel size of $k$ equals that the input fea-tures are first shifted according to the $k$ kernel positions, and then processed bya point-wise convolution (Conv 1). Motivated by this, we can learn an input-based mask, which will be used to zero and re-weight the shifted features. Then,using a Conv 1 to process the masked shifted features will equal a fully dynamicconvolution shown in Fig 2 (b). A light-weight module, $\varPsi ( \cdot )$ , and a non-linearactivation, $\phi ( \cdot )$ can be used to generate the mask, where the activation func-tion can be the one whose output is always 0 with any negative inputs (such asReLU [1] or a general-restricted tanh function [44]).

Formally, suppose we have a standard convolution with the kernel $\kappa \in$$\mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times k }$ , where $k$ is the kernel size and $C _ { \mathrm { o u t } }$ , $C _ { \mathrm { i n } }$ are the numbers of inputand output channels. Given an input $\pmb { f } \in \mathbb { R } ^ { C _ { \mathrm { i n } } \times T }$ ( $T$ is the length of temporaldimension and $f _ { t } \in \mathbb { R } ^ { C _ { \mathrm { i n } } }$ , is the tensor at time $t$ ), then we can shift the input by

$$
\hat {\boldsymbol {f}} ^ {(k)} = \operatorname {S h i f t} (\boldsymbol {f}, k) = \left[ \hat {\boldsymbol {f}} ^ {0}; \dots ; \hat {\boldsymbol {f}} ^ {k - 1} \right] \in \mathbb {R} ^ {k C _ {\mathrm {i n}} \times T}, \tag {1}
$$

$$
\hat {f} _ {t} ^ {s} = f _ {t - \lfloor k / 2 \rfloor + s}, s = 0, 1, \dots , k - 1, t = 1, 2, \dots , T,
$$

where $\hat { \pmb f } ^ { s } \in \mathbb { R } ^ { C _ { \mathrm { i n } } \times T }$ , the empty positions of the shifted features will be paddedby all-zero tensors. The weighted masks can be calculated by

$$
\boldsymbol {M} = \phi (\Psi (\boldsymbol {f})) ， \boldsymbol {M} \in \mathbb {R} ^ {C _ {\mathrm {m}} \times T}, \tag {2}
$$

where $C _ { \mathrm { m } }$ and $\psi$ can be designed into different formations to achieve differentdynamic properties. By repeated unpsampling the M to the dimension of fˆ(k) $M$ $\hat { \pmb f } ^ { ( k ) }$(denoted by ↑ (·)), we have

$$
\bar {\boldsymbol {f}} = \uparrow (\boldsymbol {M}) \odot \hat {\boldsymbol {f}} ^ {(k)} = \overline {{\boldsymbol {M}}} \odot \hat {\boldsymbol {f}} ^ {(k)}, \tag {3}
$$

where we use $M$ to represent $\uparrow ( M )$ for simplicity, and $\odot$ means element-wisemultiplication. The final output features can be written as

$$
\boldsymbol {y} = \operatorname {D F A} (\boldsymbol {f}) = \sum_ {s = 0} ^ {k - 1} \mathcal {K} _ {s} \left(\bar {\boldsymbol {M}} _ {[ s C _ {\mathrm {i n}} + 1: (s + 1) C _ {\mathrm {i n}} ]} \odot \hat {\boldsymbol {f}} ^ {s}\right), \tag {4}
$$

where $\boldsymbol { K } _ { s } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } }$ , $\overline { { M } } _ { [ ( s C _ { \mathrm { i n } } + 1 : ( s + 1 ) C _ { \mathrm { i n } } ] } \ \in \ \mathbb { R } ^ { C _ { \mathrm { i n } } \times T } .$ means the mask tensorwith the index from $s C _ { \mathrm { i n } } + 1$ to $( s + 1 ) C _ { \mathrm { i n } }$ , and $\pmb { y } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times T }$ is the output.

Different formations of DFA. Using different $\varPsi ( \cdot )$ will result in the differentformations of DFA. By implementing $\varPsi ( \cdot )$ as a convolution, the DFA can be builtinto a convolution-based DFA module (DFA_Conv). Moreover, changing the $C _ { \mathrm { m } }$leads to the different dynamic properties. Take DFA_Conv as an example. K-formation: using $C _ { \mathrm { m } } ~ = ~ k$ , the $\uparrow$ (·) will repeat the mask in the interleavedmanner at the channel dimensions with $C _ { \mathrm { i n } }$ times, which makes the DFA sharethe same dynamic receptive field among the channel dimension. C-formation:For $C _ { \mathrm { m } } = C _ { \mathrm { i n } }$ , the $\uparrow ( \cdot )$ will repeat the mask $k$ times at the channel dimensions,which makes the DFA a temporal dynamic channel pruning convolution. Also,using $C _ { \mathrm { m } } ~ { = } ~ k C _ { \mathrm { i n } }$ (CK), the DFA will adapt the receptive fields at differenttimestamps and channels.

Moreover, inspired by the success of Transformer-based architecture, we fur-ther implemented the DFA in as the formation in Fig. 3 (b). The generated maskwill result in the zeros in the attention matrix at the unimportant timestamps,and then the masked attentions will be used to re-weight the shifted features.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/25e586e889df7d897c069b3dade443b10dd3e7a1c717c30f17c21c46d51cfb66.jpg)



Fig. 3: Two different formations of the proposed DFA module.


Differences to existing works. If we restrict that $M \in \{ 0 , 1 \}$ has the samenumber of masked positions, then DFA will equal to 1-d deformable convolu-tion [12, 24], which adapts the shape of the kernels at different timestamp. Ifwe remove the $\phi ( \cdot )$ , the DFA will equal to the dynamic convolution in [9, 58]which only adapts the kernel weights-based inputs. Moreover, TD3DConv in [25]first uses temporal attention weights to adjust the features and then uses DC

for feature learning, where these weights(W) are input-dependent yet temporalstatic during generating new features. While these weights are fully dynamic inour DFA. The DFA provides an effective way for dynamic feature aggregation,which addresses the aforementioned issues in modern TAD models. Therefore,based on it, we further develop two important components Dynamic encoderand Dynamic TAD head for the proposed TAD model, DyFADet.

# 3.2 DyFADet

Overview Based on the dynamic property of DFA, we can build a TAD modelthat can effectively extract the discriminative representations in the encoderand adapt the detection heads for the action instances with different ranges.Following the common design in [42, 62], the proposed DyFADet consists of avideo feature backbone, an encoder, and two detection heads for action classi-fication and localization. Concretely, we first extract the video features usinga pre-trained backbone. Then the extracted representations will be sent to theencoder to generate the features pyramid, where the former features will bedown-sampled by the DynE layer with the scale of 2 to obtain $f ^ { l } , l = 1 , . . . , L$ ( $L$is the number of total levels). These features will be then used by DyHead foraction detection. The architecture of DyFADet is shown in Fig. 4 (a), where theDynE layer and DyHead are two components built based on DFA applying thedynamic feature learning strategy.

Feature encoder with DynE. As illustrated in Fig. 4 (b), the DynE layer isbuilt by substituting the SA module with the proposed DynE module, which fol-lows the marco-architecture of transformer layers [62]. The DynE module has twobranches: A instance-dynamic branch based on DFA_Conv with kernel size of 1generates the global temporal weighted mask to help aggregate global the actioninformation. For another branch, we propose to use the DFA_Conv with con-volutions implemented by different kernel sizes to better generate the weightedmask tenors. To further improve the efficiency during feature learning, all con-volution modules are implemented by depth-wise convolutions (D Conv) withthe corresponding dimensionality. The two branches can be written as

$$
\boldsymbol {f} _ {\text {i n s}} = \operatorname {D F A} _ {-} \operatorname {C o n v} _ {1} (\operatorname {S q u e e z e} (\operatorname {L N} (\operatorname {D S} (\boldsymbol {f})))) \tag {5}
$$

$$
\boldsymbol {f} _ {\mathrm {k}} = \operatorname {D F A} _ {-} \operatorname {C o n v} _ {k, w} (\operatorname {L N} (\operatorname {D S} (\boldsymbol {f}))), \tag {6}
$$

where the DS is the down-sampling achieved by max-pooling with the scaleof 2. LN is the Layer normalization [3]. The Squeeze is achieved by averagepooling at the channel dimension. $w$ is the factor to expand the window size ofthe convolution to $w * ( k + 1 )$ , which enables the module can learn a the featurewith long-term information. Then the output features of the DynE module canbe represented by

$$
\boldsymbol {f} _ {\text {d y n}} = \boldsymbol {f} _ {\mathrm {w}} + \boldsymbol {f} _ {\text {i n s}} + \operatorname {D S} (\boldsymbol {f}). \tag {7}
$$

Overall, in the feature encoder, each DynE layer will down-sample the fea-ture with a scale of 2 to generate the representations with different temporalresolutions. The dynamic feature selection ability of DynE layers will guarantee

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/081ecf2342503f47e0fa0e098a2e2cd98c53685b45909a56d0ae7ba07bc30bde.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/b915e601eb490abeaff203805020c958204604a22c122a36d52d08ab903e3a02.jpg)



(b) The DynE layer.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/0acff8509af9c1b600a52a13c4388c8e520208303849dffd6f96e3175acbb860.jpg)



(c)Feature fusion in MSDy-head.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/e9f3de8e98a4e5f2327a628bad0ec369f5e6ec1bf1b5a9299ec4b20ef651eb96.jpg)



(d) Cls and Reg modules.



Fig. 4: (a) Overview of DyFADet. (b) The DynE layer consisting of the feature encoder.GN is Group-normalization [54]. (c) The multi-scale feature fusion in DyHead. (d) Theclassification and regression module obtains the classification and boundary results.


the discriminative information of the obtained representations and alleviate theless-discriminant feature problem, which leads to better TAD performance.

Dynamic TAD head. The shared-weight static detection heads in commonTAD models can have the head inadaptation issue, which means that the optimalweights for detecting long-range and short-range action instances can be differ-ent, resulting in the sub-optimal detection performance. Even when implementedwith the cross-level feature fusion, the heads only show limited improvement. In-tuitively, fusing the features from a higher scale helps the head to explore moreglobal long-term information. Exploring the information from a lower scale canenable the head to find more boundary details. We infer that such a multi-scalefusion can benefit the detection performance only if these intrinsic representa-tions are properly selected from the adjacent scale. These motivate us to build anovel DFA, which can dynamically adjust the shared parameters based on inputsand selectively fuse the cross-level features for better performance.

The architecture of a DyHead is illustrated in Fig. 4 (a). Both the featuresfrom the corresponding and the adjacent levels in the pyramid are sent to theDyHead. By implementing the dynamic feature learning mechanism with theproposed DFA, the head parameters can adjust based on inputs, and importantrepresentations will be selectively fused. Specifically, suppose that the depth ofthe head is $D$ , the output features $\pmb { f } _ { d + 1 } ^ { l }$ at the $\it l$ -th level can be calculated asthe accumulation of $f _ { d } ^ { l - 1 }$ , $\pmb { f } _ { d } ^ { l }$ and $\pmb { f } _ { d } ^ { l + 1 }$ , which are processed by three differentpaths (shown in Fig. 4 (c)). The down path and up path are built based on

the DFA with an additional LN and down-sampling (DS) or up-sampling (US)interpolation module with a scale of 2. The features will be first fused by

$$
\begin{array}{l} \tilde {\boldsymbol {f}} _ {d} ^ {l} = \gamma_ {d} \cdot \left(\mathrm {D S} (\text {D F A} _ {-} \operatorname {A t t} (\boldsymbol {f} _ {d} ^ {l - 1})) + \right. \\ \left. \operatorname {U S} \left(\operatorname {D F A} _ {-} \operatorname {A t t} \left(\boldsymbol {f} _ {d} ^ {l + 1}\right)\right)\right) + \alpha_ {d} \cdot \boldsymbol {f} _ {d} ^ {l}, \tag {8} \\ \end{array}
$$

where $\gamma _ { d }$ and $\alpha _ { d }$ are two learnable factors. Then the resultant feature will beprocessed by the depth path by

$$
\boldsymbol {f} _ {d + 1} ^ {l} = \operatorname {D F A} _ {-} \operatorname {A t t} \left(\tilde {\boldsymbol {f}} _ {d} ^ {l}\right). \tag {9}
$$

The proposed DyHead With the multi-scale dynamic feature fusion proce-dure, the final features at depth $D$ , namely, $\mathbf { \Delta } f _ { \mathbf { \delta } \mathcal { D } } ^ { l }$ , $l = 1 , . . . , L$ , will be used todetect the action instances with different temporal ranges.

Classification and regression modules. The final classification (Cls) andregression (Reg) modules are designed to process final features across all levels.The cls module is realized by using a 1D convolution with a Sigmoid function topredict the probability of action categories at each time stamp. The Reg moduleis implemented using a 1D convolution with a ReLU to estimate the distances$\{ d _ { m } ^ { s } , d _ { m } ^ { e } \}$ from the timestamp $t$ to the action start and end $\{ t _ { m } ^ { s } , t _ { m } ^ { e } \}$ for $m$ -thaction instance, which can be obtained by $t _ { m } ^ { s } = t - d _ { m } ^ { s }$ and $t _ { m } ^ { e } = t + d _ { m } ^ { e }$ .

# 3.3 Training and Inference

Training. Follow [62], the center sampling strategy [62] is used during training,which means that the instants around the center of an action instance are labeledas positive during training. The loss function of the proposed DyFADet followthe simple design in [62], which has two different terms: (1) ${ \mathcal { L } } _ { \mathrm { c l s } }$ is a focal loss [30]for classification; (2) $\mathcal { L } _ { \mathrm { r e g } }$ is a DIoU loss [66] for distance regression. The lossfunction is then defined as

$$
\mathcal {L} = \sum_ {t, l} \left(\mathcal {L} _ {c l s} + \lambda_ {\text {r e g}} \mathbb {I} _ {c t} \mathcal {L} _ {r e g}\right) / T _ {p o s}, \tag {10}
$$

where $T _ { p o s }$ is the number of positive timestamps and ${ \mathbb { I } } _ { c t }$ is an indicator functionthat denotes if a time step $t$ is within the sampling center of an action instance.

Inference. During inference, if a classification score is higher than the detectionthreshold, the instant will be kept. Then, Soft-NMS [4] will be further appliedto remove the repeated detected instances.

# 4 Experiments

# 4.1 Experimental settings

Datasets. Six TAD datasets, including HACS-Segment [64], THUMOS14 [23],ActivityNet-1.3 [5], Epic-Kitchen 100 [13], Ego4D-Moment Queries v1.0 (Ego4D-MQ1.0) [15] and FineAction [33], are used in our experiments. ActivityNet-1.3and HACS are two large-scale datasets with 200 classes of action, containing10,024 and 37,613 videos for training, 4,926 and 5,981 videos for tests. THU-MOS14 consists of 20 sport action classes and contains 200 and 213 untrimmedvideos with 3,007 and 3,358 action instances on the training and test set, respec-tively. The Epic-Kitchen 100 and Ego4D-MQ1.0 are two datasets in first-personvision. Epic-Kitchen 100 has two sub-tasks: noun and verb localization, con-taining 495 and 138 videos with 67,217 and 9,668 action instances for trainingand testing, respectively. Ego4D-MQ1.0 has 2,488 video clips and 22.2K actioninstances from 110 pre-defined action categories, which are densely labeled. Fine-Action contains 103K temporal instances of 106 fine-grained action categories,annotated in 17K untrimmed videos.

Evaluation metric and experimental implementation. The standard meanaverage precision (mAP) at different temporal intersection over union (tIoU)thresholds will be reported as evaluation metric in the experiments. We followthe practice in [42,62] that uses off-the-shelf pre-extracted features as input. Ourmethod is trained with AdamW [35] with warming-up. More training details areprovided in the supplementary materials.

Detailed architecture of DyFADet. We used 2 convolutions for feature em-bedding, 7 DynE layers as the encoder, and separate DyHeads for classificationand regression as the detectors. In our experiments, we report the best resultsof DyFADet with different architectures. A comprehensive ablation study aboutthe architecture is also provided in Section 4.3.

# 4.2 Main results

HACS. The performance ofdifferent TAD methods on theHACS dataset is provided in Ta-ble 1, where average mAP in[0.5:0.05:0.95] is reported, and thebest and the second best per-formance are denoted by boldand blue. In our experiments, theSlowFast [14] features are used forthe proposed DyFADet in TADtasks on HACS.

From the results, we see thatour method with SlowFast fea-tures outperforms all other eval-


Table 1: Results on HACS-segment.


<table><tr><td>Method</td><td>Backbone</td><td>0.5</td><td>0.75</td><td>0.95</td><td>Avg.</td></tr><tr><td>SSN [65]</td><td>I3D</td><td>28.8</td><td>18.8</td><td>5.3</td><td>19.0</td></tr><tr><td>LoFi [56]</td><td>TSM</td><td>37.8</td><td>24.4</td><td>7.3</td><td>24.6</td></tr><tr><td>G-TAD [57]</td><td>I3D</td><td>41.1</td><td>27.6</td><td>8.3</td><td>27.5</td></tr><tr><td>TadTR [32]</td><td>I3D</td><td>47.1</td><td>32.1</td><td>10.9</td><td>32.1</td></tr><tr><td>BMN [28]</td><td>SlowFast</td><td>52.5</td><td>36.4</td><td>10.4</td><td>35.8</td></tr><tr><td>ActionFormer [62]</td><td>SlowFast</td><td>54.9</td><td>36.9</td><td>9.5</td><td>36.4</td></tr><tr><td>TALLFormer [10]</td><td>Swin</td><td>55.0</td><td>36.1</td><td>11.8</td><td>36.5</td></tr><tr><td>TCANet [39]</td><td>SlowFast</td><td>54.1</td><td>37.2</td><td>11.3</td><td>36.8</td></tr><tr><td>TriDet [42]</td><td>SlowFast</td><td>56.7</td><td>39.3</td><td>11.7</td><td>38.6</td></tr><tr><td>TriDet [42]</td><td>VM2-g</td><td>62.4</td><td>44.1</td><td>13.1</td><td>43.1</td></tr><tr><td>Ours</td><td>SlowFast</td><td>57.8</td><td>39.8</td><td>11.8</td><td>39.2</td></tr><tr><td>Ours</td><td>VM2-g</td><td>64.0</td><td>44.8</td><td>14.1</td><td>44.3</td></tr></table>

uated methods in terms of average mAP (39.2%), and also achieves the best


Table 2: Comparison with the SOTA methods on THUMOS14 and ActivityNet-1.3.TSN [49], I3D [7], Swin Transformer (Swin) [10] and TSP (R(2+1)D) [2] features areused. The best and the second best performance are denoted by bold and blue.


<table><tr><td rowspan="2">Method</td><td colspan="7">THUMOS14</td><td colspan="4">ActivityNet-1.3</td></tr><tr><td>Features</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>Avg.</td><td>Features</td><td>0.5</td><td>0.75</td><td>0.95 Avg.</td></tr><tr><td>BMN [28]</td><td>TSN</td><td>56.0</td><td>47.4</td><td>38.8</td><td>29.7</td><td>20.5</td><td>38.5</td><td>TSN</td><td>50.1</td><td>34.8</td><td>8.3 33.9</td></tr><tr><td>G-TAD [57]</td><td>TSN</td><td>54.5</td><td>47.6</td><td>40.3</td><td>30.8</td><td>23.4</td><td>39.3</td><td>TSN</td><td>50.4</td><td>34.6</td><td>9.0 34.1</td></tr><tr><td>A2Net [60]</td><td>I3D</td><td>58.6</td><td>54.1</td><td>45.5</td><td>32.5</td><td>17.2</td><td>41.6</td><td>I3D</td><td>43.6</td><td>28.7</td><td>3.7 27.8</td></tr><tr><td>PBRNet [31]</td><td>I3D</td><td>58.5</td><td>54.6</td><td>51.3</td><td>41.8</td><td>29.5</td><td>-</td><td>I3D</td><td>54.0</td><td>35.0</td><td>9.0 35.0</td></tr><tr><td>TCANet [39]</td><td>TSN</td><td>60.6</td><td>53.2</td><td>44.6</td><td>36.8</td><td>26.7</td><td>44.3</td><td>TSN</td><td>52.3</td><td>36.7</td><td>6.9 35.5</td></tr><tr><td>RTD-Net [46]</td><td>I3D</td><td>68.3</td><td>62.3</td><td>51.9</td><td>38.8</td><td>23.7</td><td>49.0</td><td>I3D</td><td>47.2</td><td>30.7</td><td>8.6 30.8</td></tr><tr><td>VSGN [63]</td><td>TSN</td><td>66.7</td><td>60.4</td><td>52.4</td><td>41.0</td><td>30.4</td><td>50.2</td><td>I3D</td><td>52.3</td><td>35.2</td><td>8.3 34.7</td></tr><tr><td>AFSD [26]</td><td>I3D</td><td>67.3</td><td>62.4</td><td>55.5</td><td>43.7</td><td>31.1</td><td>52.0</td><td>I3D</td><td>52.4</td><td>35.2</td><td>6.5 34.3</td></tr><tr><td>ReAct [43]</td><td>TSN</td><td>69.2</td><td>65.0</td><td>57.1</td><td>47.8</td><td>35.6</td><td>55.0</td><td>TSN</td><td>49.6</td><td>33.0</td><td>8.6 32.6</td></tr><tr><td>TadTR [32]</td><td>I3D</td><td>74.8</td><td>69.1</td><td>60.1</td><td>46.6</td><td>32.8</td><td>56.7</td><td>TSN</td><td>51.3</td><td>35.0</td><td>9.5 34.6</td></tr><tr><td>TALLFormer [10]</td><td>Swin</td><td>76.0</td><td>-</td><td>63.2</td><td>-</td><td>34.5</td><td>59.2</td><td>Swin</td><td>54.1</td><td>36.2</td><td>7.9 35.6</td></tr><tr><td>ActionFormer [62]</td><td>I3D</td><td>82.1</td><td>77.8</td><td>71.0</td><td>59.4</td><td>43.9</td><td>66.8</td><td>R(2+1)D</td><td>54.7</td><td>37.8</td><td>8.4 36.6</td></tr><tr><td>ASL [41]</td><td>I3D</td><td>83.1</td><td>79.0</td><td>71.7</td><td>59.7</td><td>45.8</td><td>67.9</td><td>I3D</td><td>54.1</td><td>37.4</td><td>8.0 36.2</td></tr><tr><td>TriDet [42]</td><td>I3D</td><td>83.6</td><td>80.1</td><td>72.9</td><td>62.4</td><td>47.4</td><td>69.3</td><td>R(2+1)D</td><td>54.7</td><td>38.0</td><td>8.4 36.8</td></tr><tr><td>Ours</td><td>I3D</td><td>84.0</td><td>80.1</td><td>72.7</td><td>61.1</td><td>47.9</td><td>69.2</td><td>R(2+1)D</td><td>58.1</td><td>39.6</td><td>8.4 38.5</td></tr></table>

performance across all tIoUs thresholds. Notably, with tIoU = 0.5, the DyFADetsurpasses the Tridet by 1.1%. As a TAD model can generally benefit from a moreadvanced backbone, we further implement VideoMAE V2-Gaint [48] (VM2-g) toconduct TAD on HACS. Remarkably, our method achieves the highest perfor-mance with VideoMAE V2-Gaint and beats the previous SOTA TriDet (VM2-g)with the new SOTA result on HACS, $4 4 . 3 \%$ .

THUMOS14 and ActivityNet-1.3. The experimental results which com-pare the performance of ours and other TAD models are shown in Table 2. Av-erage mAP in [0.3:0.1:0.7] and [0.5:0.05:0.95] are reported on THUMOS14 andActivityNet-1.3, respectively. In the experiments, our method conducts the TADtask based on I3D and R(2+1)D features for the THUMOS14 and ActivityNet-1.3 datasets. We see that our method with I3D features achieves the mAP of$6 9 . 2 \%$ , which is competitive to TriDet [42], while significantly outperforming allother related TAD methods on THUMOS14. On ActivityNet-1.3, the proposedDyFADet significantly surpasses the TriDet by about 1.7%, and achieves themAP of 38.5%. The high performance of the proposed DyFADet indicates theeffectiveness of dynamic feature learning mechanisms in modern TAD methods.

Generally, a TAD model canbenefit from a more advancedbackbone. Therefore, we fur-ther implement VideoMAE V2-Gaint [48] (VM2-g) to conductTAD on THUMOS14. We seethat all TAD methods achieve


Table 3: Results on THUMOS14 using VM2-g.


<table><tr><td>Method</td><td>Backbone</td><td>0.3</td><td>0.55</td><td>0.7</td><td>Avg.</td></tr><tr><td>ActionFormer [62]</td><td>VM2-g</td><td>84.0</td><td>73.0</td><td>47.7</td><td>69.6</td></tr><tr><td>TriDet [42]</td><td>VM2-g</td><td>84.8</td><td>73.3</td><td>48.8</td><td>70.1</td></tr><tr><td>Ours</td><td>VM2-g</td><td>84.3</td><td>73.7</td><td>50.2</td><td>70.5</td></tr><tr><td>Ours</td><td>VM2-g+F</td><td>85.4</td><td>74.0</td><td>50.2</td><td>71.1</td></tr></table>

significant improvements with an advanced feature extraction backbone, VM2-g. While, our method can achieve the mAP of $7 0 . 5 \%$ , which is superior to TriDetand ActionFormer. Also, the detection performance can be boosted to 71.1% ifwe additionally use the optic flow (F) features.


Table 4: Results on Epic-Kitchen 100.


<table><tr><td></td><td>Method</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>Avg.</td></tr><tr><td rowspan="3">V.</td><td>G-TAD [57]</td><td>12.1</td><td>11.0</td><td>9.4</td><td>8.1</td><td>6.5</td><td>9.4</td></tr><tr><td>ActionFormer [62]</td><td>26.6</td><td>25.4</td><td>24.2</td><td>22.3</td><td>19.1</td><td>23.5</td></tr><tr><td>ASL [41]</td><td>27.9</td><td>-</td><td>25.5</td><td>-</td><td>19.8</td><td>24.6</td></tr><tr><td></td><td>Ours</td><td>28.0</td><td>27.0</td><td>25.6</td><td>23.5</td><td>20.8</td><td>25.0</td></tr><tr><td rowspan="3">N.</td><td>G-TAD [57]</td><td>11.0</td><td>10.0</td><td>8.6</td><td>7.0</td><td>5.4</td><td>8.4</td></tr><tr><td>ActionFormer [62]</td><td>25.2</td><td>24.1</td><td>22.7</td><td>20.5</td><td>17.0</td><td>21.9</td></tr><tr><td>ASL [41]</td><td>26.0</td><td>-</td><td>23.4</td><td>-</td><td>17.7</td><td>22.6</td></tr><tr><td></td><td>Ours</td><td>26.8</td><td>26.0</td><td>24.1</td><td>21.9</td><td>18.5</td><td>23.4</td></tr></table>


Table 5: Results on Ego4D-MQ1.0.


<table><tr><td>Method</td><td>Features</td><td>0.1</td><td>0.3</td><td>0.5</td><td>Avg.</td></tr><tr><td>VSGN [63]</td><td>SF</td><td>9.1</td><td>5.8</td><td>3.4</td><td>6.0</td></tr><tr><td>VSGN [27]</td><td>EV</td><td>16.6</td><td>11.5</td><td>6.6</td><td>11.4</td></tr><tr><td>ActionFormer [62]</td><td>SF</td><td>20.1</td><td>14.4</td><td>10.0</td><td>14.9</td></tr><tr><td>ActionFormer [62]</td><td>EV</td><td>26.9</td><td>20.2</td><td>13.7</td><td>20.3</td></tr><tr><td>ActionFormer [62]</td><td>EV+SF</td><td>28.0</td><td>21.2</td><td>15.6</td><td>21.6</td></tr><tr><td>Ours</td><td>SF</td><td>19.0</td><td>15.0</td><td>11.2</td><td>15.3</td></tr><tr><td>Ours</td><td>EV</td><td>28.4</td><td>22.1</td><td>16.2</td><td>22.2</td></tr><tr><td>Ours</td><td>EV+SF</td><td>28.8</td><td>22.6</td><td>16.9</td><td>22.8</td></tr></table>

Epic-Kitchen 100 and Ego4D-MQ1.0. The evaluations are also conductedon two large-scale egocentric datasets, which are shown in Table 4 and Table 5,respectively. For Epic-Kitchen 100, the average mAP in [0.1:0.1:0.5] is reportedand all methods use the SlowFast features. We see that our method has the bestperformance across all tIoU thresholds on both subsets and achieves an averagemAP of $2 5 . 0 \%$ and $2 3 . 4 \%$ for verb and noun subsets, respectively, which aresignificantly superior to the strong performance of the recent TAD methods, in-cluding Actionformer [62], ASL [41]. For Ego4D-MQ1.0, two types of features,including SlowFast (SF) and EgoVLP (EV) features are used in the experiments.With SlowFast features, the proposed method achieves the mAP of $1 5 . 3 \%$ , whichsignificantly outperforms the Actionformer. Moreover, we see that using or com-bining the features with advanced backbone models, such as EgoVLP [27], canfurther boost the performance of our method by a large margin. SF and EVdenote Slowfast [14] and EgoVLP [27] features. V. and N. denote the verb andnoun sub-tasks.

FineAction. In the experiments,we report the performance of thedifferent popular TAD methods in-cluding BMN [28], G-TAD [57], Ac-tionFormer [62], and our methodon TAD task. Moreover, I3D [49],InternVideo [50] and and VM2-g [48] are used to extract the off-line features and the average mAPin [0.50:0.05:0.95] is reported for allmethods. The experimental results


Table 6: Results on FineAction.


<table><tr><td>Method</td><td>Backbone</td><td>0.5</td><td>0.75</td><td>0.95</td><td>Avg.</td></tr><tr><td>BMN</td><td>I3D</td><td>14.4</td><td>8.9</td><td>3.1</td><td>9.3</td></tr><tr><td>G-TAD</td><td>I3D</td><td>13.7</td><td>8.8</td><td>3.1</td><td>9.1</td></tr><tr><td>ActionFormer</td><td>InternVideo</td><td>-</td><td>-</td><td>-</td><td>17.6</td></tr><tr><td>ActionFormer</td><td>VM2-g</td><td>29.1</td><td>17.7</td><td>5.1</td><td>18.2</td></tr><tr><td>Ours</td><td>VM2-g</td><td>37.1</td><td>23.7</td><td>5.9</td><td>23.8</td></tr></table>

are provided in Tab.6. The experimental results show that our method outper-forms other TAD models and achieve a new SOTA result on FineAction, 23.8%,with the VideoMAEv2-giant.

# 4.3 Ablation study

Ablation studies are conducted on THUMOS14 to explore more properties aboutthe DFA and DyFADet.

DFA modules in TAD. The ex-periments investigate the effectivenessof different dynamic feature aggrega-tion modules in TAD tasks. The re-sults are shown in Table 7, where dif-ferent implementations of the encoderand the detection head are evalu-ated. The baseline model is realized byan all-convolution TAD model, whichachieved the mAP of $6 2 . 1 \%$ . We fur-ther use different dynamic modules tosubstitute or improve the convolutionsin the encoder as the comparison, suchas the deformable 1-d convolution [24],

squeeze-and-excitation module [20], Dynamic convolution [67] and temporal de-formable 3d convolution module [25]. All dynamic modules achieve better per-formance than the baseline model with convolution, indicating the strong abilityof dynamic modules in TAD tasks. While, due to the stronger adaptation abilityof DFA, the DyFADet∗ substituting the convolutions with DFA can achieve theperformance equaling to the recent strong TAD model, Actionformer. Moreover,using the proposed DynE layer (DyFADet†) further increases the final perfor-mance by 1.0%. For the detection head, we see that applying the multi-scaleconnection in the TAD head can improve the final detection performance. How-ever, naively using the convolution to connect different scales (DyFADet‡) onlyresults in limited improvements. While, after being equipped with DyHead, theDyFADet can achieve a performance of 69.2%, outperforming all other mod-els in the experiments. A more comprehensive study about the architecture forDyFADet is provided in supplementary materials.

The discriminability of the learned features. As shown in [42,47], the fea-tures obtained by recent TAD methods [52, 62] tend to exhibit high similaritiesbetween snippets, which leads to the less-discriminant feature problem and beharmful to TAD performance. In Fig. 5(a), we perform statistics of the aver-age cosine similarity between each feature at different timestamps. We observethat the features in Actionformer exhibit high similarity, indicating poor dis-criminability. Using the DynE based encoder can address the issue and thereforeimprove the detection performance. Moreover, we further provide the featuresimilarity matrix between the different timestamps in Fig.5(b), where the redboxes exhibit the action intervals. The darker color means the features are morediscriminant and share less similarity. From the result, we see that the inter-classfeatures from our method within the red boxes show strong similarities, resulting


Table 7: Results with different modules.


<table><tr><td>Method</td><td>Encoder</td><td>MS-head</td><td>Avg.</td></tr><tr><td>Baseline</td><td>Conv</td><td>×</td><td>62.1</td></tr><tr><td>Baseline</td><td>DeformConv [24]</td><td>×</td><td>66.1</td></tr><tr><td>Baseline</td><td>SE [20]</td><td>×</td><td>63.4</td></tr><tr><td>Baseline</td><td>Dyn Conv [67]</td><td>×</td><td>66.7</td></tr><tr><td>Baseline</td><td>TD3d Conv [25]</td><td>×</td><td>66.5</td></tr><tr><td>DyFADet*</td><td>DFA_Conv</td><td>×</td><td>66.8</td></tr><tr><td>ActionFormer</td><td>SA</td><td>×</td><td>66.8</td></tr><tr><td>DyFADet†</td><td>DynE</td><td>×</td><td>67.8</td></tr><tr><td>DyFADet‡</td><td>Conv</td><td>Dyn</td><td>67.9</td></tr><tr><td>DyFADet‡</td><td>DynE</td><td>Conv</td><td>68.0</td></tr><tr><td>DyFADet</td><td>DynE</td><td>Dyn</td><td>69.2</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/951131b410bc6f2da29094e55812f27c15cbd4df2b574994ec95889066e3bfde.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/5473a419c7751fa8600f3edc13fb8faf984943cec112082eab032aa6d8a6ae15.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/9fcb143c13108184ede0afb1a56937d362eb3450a5d365165b28a7882c30ae49.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/7837e26128c0968977bcf1315c1c5003b4a7e3d7e07a2d2993292ab87c354c12.jpg)



Fig. 5: (a) The average cosine similarity between features at different timestamps inthe same level among each encoder layer. (b) Similarity matrix of the extracted featuresamong timestamps.


in that the boundary features are distinctive and can be easily extracted. WhileActionformer fails to explore the discriminant features during feature learning.From the results, our method addresses the mentioned first issue in Fig.1 basedon the more discriminant learned features in DyFADet $^ \dagger$ , which can be furtherenhanced by using the proposed DyFADet as TAD model.

Visualization of DFA in DyHead. In Fig. 6 (a), we provide the visualizationresults of the proposed DFA. For better visualization, we use the DyFADet with$C _ { m } = k$ in Eq.(2), meaning that the same receptive fields will be shared amongchannels while varying at different timestamps. From the results we see that theDFA can adapt the re-weight masks based on inputs, leading to different for-mations at different timestamps. Noting that in classical TAD models, such asActionformer [62], the parameters of the detection head are shared among differ-ent levels, which might be harmful in detection. While, the dynamic propertiesand multi-scale fusion of the DyFADet specialize the detection head for inputsand target action instances, leading to a fine-grained dynamic action detectionmanner which can achieve better TAD performance.

Moreover, we visualize the average activated rate at each path for the wholevideo in DyHead (shown in Fig. 6 (b)). The results show that the proposedDyHead achieves the dynamic routing-like feature learning mechanism similarto [44]. However, our method can simultaneously adapt the receptive fields andthe kernel weights, which improves the model ability in TAD tasks. More visu-alization results can be found in supplementary materials.


Table 8: Different hyper-parameters.


<table><tr><td>w</td><td>mAP</td><td>D</td><td>mAP</td><td>DynType</td><td>mAP</td></tr><tr><td>3</td><td>68.2</td><td>2</td><td>67.4</td><td>C</td><td>68.4</td></tr><tr><td>5</td><td>69.2</td><td>3</td><td>69.2</td><td>K</td><td>69.2</td></tr><tr><td>6</td><td>67.0</td><td>4</td><td>67.1</td><td>CK</td><td>68.4</td></tr></table>


Table 9: Average latency on THUMOS14.


<table><tr><td rowspan="2">Method</td><td colspan="2">AP</td><td colspan="2">Latency</td><td>Params</td></tr><tr><td>0.5</td><td>Avg.</td><td>CPU(s)</td><td>GPU(ms)</td><td>(M)</td></tr><tr><td>ActionFormer</td><td>71.0</td><td>66.8</td><td>0.57</td><td>68.4</td><td>29.2</td></tr><tr><td>DyFADet†</td><td>71.7</td><td>67.8</td><td>0.16</td><td>44.7</td><td>12.7</td></tr><tr><td>DyFADet</td><td>72.7</td><td>69.2</td><td>0.49</td><td>58.7</td><td>18.8</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/2fd09604e2efa2711b6a4e8a2dc28922c91676769cb8db053840199ada4912db.jpg)



(a) Visualization of reweighted masks at different timestamps at Level-3.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/804aac19e9dd1bc515bab511379a47693f0f10e54c19ac9b26c1a43d5fce014a.jpg)



(b)Average path strength during inference.



Fig. 6: Visualization when DyFADet detecting action High Jump. (a) the DFAreweight masks at different timestamps at Level-3 in DyHead. (b) Average pathstrength during inference of the whole video.


Hyper-parameters. In Table 8 , we further evaluate the performance of Dy-FADet on THUMOS14 with different hyper-parameters, including the expandedfactor, $w$ , the layer number of the detection head, and the dynamic type of theproposed module. We observe the model works best with $D = 3$ and the w shouldbe selected for different datasets. Moreover, although the dynamic type can af-fect the final performance, DyFADet with each dynamic type achieves higherperformance compared to most TAD methods in Table 2.

Latency. We test the average latency for single video inference on GeForce RTX4090 GPU on THUMOS14. As shown in Table 9, DyFADet† is faster than Ac-tionFormer while has better detection performance. Moreover, DyFADet can beimproved by the DyHead which further brings 1.4% average mAP improvementand the latency is still comparable to Actionformer.

# 5 Conclusion

In this paper, we introduced a novel DFA module simultaneously adapting itskernel weights and receptive fields, to address the less-discriminant feature andhead inadaptation issues in TAD models. The proposed DyFADet based on DFAachieves high performance on a series of TAD benchmarks, which indicates thatan input-based fine-grained feature extraction mechanism should be consideredfor building high-performance TAD models. For future works, we believe thatthe efficiency of DyFADet can be further improved by combining the sparse con-volution [11] and adding additional constraints to encourage each DFA to maskas many features as possible with a minor performance penalty. The applicationsof DFA in more video-understanding tasks will be further investigated.

Acknowledgement. This work is supported in part by National Natural Sci-ence Foundation of China under Grants 62206215, China Postdoctoral ScienceFoundation under Grants 2022M712537, and China National Postdoctoral Pro-gram for Innovative Talents BX2021241.

# References



1. Agarap, A.F.: Deep learning using rectified linear units (ReLU). arXiv preprintarXiv:1803.08375 (2018) 4





2. Alwassel, H., Giancola, S., Ghanem, B.: Tsp: Temporally-sensitive pretraining ofvideo encoders for localization tasks. In: ICCV (2021) 10, 20





3. Ba, J.L., Kiros, J.R., Hinton, G.E.: Layer normalization. arXiv preprintarXiv:1607.06450 (2016) 6, 19





4. Bodla, N., Singh, B., Chellappa, R., Davis, L.S.: Soft-nms–improving object detec-tion with one line of code. In: ICCV (2017) 8, 21





5. Caba Heilbron, F., Escorcia, V., Ghanem, B., Carlos Niebles, J.: Activitynet: Alarge-scale video benchmark for human activity understanding. In: CVPR (2015)3, 9





6. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: End-to-end object detection with transformers. In: ECCV (2020) 3





7. Carreira, J., Zisserman, A.: Quo vadis, action recognition? a new model and thekinetics dataset. In: CVPR (2017) 10, 19





8. Chen, G., Zheng, Y.D., Wang, L., Lu, T.: Dcan: improving temporal action detec-tion via dual context aggregation. In: AAAI (2022) 3





9. Chen, Y., Dai, X., Liu, M., Chen, D., Yuan, L., Liu, Z.: Dynamic convolution:Attention over convolution kernels. In: CVPR (2020) 3, 5





10. Cheng, F., Bertasius, G.: Tallformer: Temporal action localization with a long-memory transformer. In: ECCV (2022) 3, 9, 10





11. Contributors, S.: Spconv: Spatially sparse convolution library. https://github.com/traveller59/spconv (2022) 14





12. Dai, J., Qi, H., Xiong, Y., Li, Y., Zhang, G., Hu, H., Wei, Y.: Deformable convo-lutional networks. In: ICCV (2017) 3, 5





13. Damen, D., Doughty, H., Farinella, G.M., Furnari, A., Kazakos, E., Ma, J., Molti-santi, D., Munro, J., Perrett, T., Price, W., et al.: Rescaling egocentric vision:Collection, pipeline and challenges for epic-kitchens-100. IJCV (2022) 3, 9





14. Feichtenhofer, C., Fan, H., Malik, J., He, K.: Slowfast networks for video recogni-tion. In: ICCV (2019) 9, 11, 20





15. Grauman, K., Westbury, A., Byrne, E., Chavis, Z., Furnari, A., Girdhar, R., Ham-burger, J., Jiang, H., Liu, M., Liu, X., et al.: Ego4d: Around the world in 3,000hours of egocentric video. In: CVPR (2022) 3, 9





16. Han, Y., Huang, G., Song, S., Yang, L., Wang, H., Wang, Y.: Dynamic neuralnetworks: A survey. IEEE TPAMI (2021) 2, 3





17. Han, Y., Liu, Z., Yuan, Z., Pu, Y., Wang, C., Song, S., Huang, G.: Latency-awareunified dynamic networks for efficient image recognition. IEEE TPAMI (2024) 3





18. Han, Y., Pu, Y., Lai, Z., Wang, C., Song, S., Cao, J., Huang, W., Deng, C.,Huang, G.: Learning to weight samples for dynamic early-exiting networks. In:ECCV (2022) 3





19. Han, Y., Yuan, Z., Pu, Y., Xue, C., Song, S., Sun, G., Huang, G.: Latency-awarespatial-wise dynamic networks. NeurIPS (2022) 3





20. Hu, J., Shen, L., Sun, G.: Squeeze-and-excitation networks. In: CVPR (2018) 12





21. Huang, G., Chen, D., Li, T., Wu, F., van der Maaten, L., Weinberger, K.: Multi-scale dense networks for resource efficient image classification. In: ICLR (2018)3





22. Huang, G., Wang, Y., Lv, K., Jiang, H., Huang, W., Qi, P., Song, S.: Glance andfocus networks for dynamic visual recognition. IEEE TPAMI (2023) 2





23. Jiang, Y.G., Liu, J., Roshan Zamir, A., Toderici, G., Laptev, I., Shah, M., Suk-thankar, R.: THUMOS challenge: Action recognition with a large number of classes.http://crcv.ucf.edu/THUMOS14/ (2014) 3, 9





24. Lei, P., Todorovic, S.: Temporal deformable residual networks for action segmen-tation in videos. In: CVPR (2018) 3, 5, 12





25. Li, J., Liu, X., Zhang, M., Wang, D.: Spatio-temporal deformable 3d convnets withattention for action recognition. PR 98, 107037 (2020) 3, 5, 12





26. Lin, C., Xu, C., Luo, D., Wang, Y., Tai, Y., Wang, C., Li, J., Huang, F., Fu, Y.:Learning salient boundary feature for anchor-free temporal action localization. In:CVPR (2021) 2, 3, 10





27. Lin, K.Q., Wang, J., Soldan, M., Wray, M., Yan, R., XU, E.Z., Gao, D., Tu, R.C.,Zhao, W., Kong, W., et al.: Egocentric video-language pretraining. NeurIPS (2022)11, 20





28. Lin, T., Liu, X., Li, X., Ding, E., Wen, S.: Bmn: Boundary-matching network fortemporal action proposal generation. In: ICCV (2019) 3, 9, 10, 11





29. Lin, T., Zhao, X., Su, H., Wang, C., Yang, M.: Bsn: Boundary sensitive networkfor temporal action proposal generation. In: ECCV (2018) 3





30. Lin, T.Y., Goyal, P., Girshick, R., He, K., Dollár, P.: Focal loss for dense objectdetection. In: ICCV (2017) 8





31. Liu, Q., Wang, Z.: Progressive boundary refinement network for temporal actiondetection. In: AAAI (2020) 10





32. Liu, X., Wang, Q., Hu, Y., Tang, X., Zhang, S., Bai, S., Bai, X.: End-to-endtemporal action detection with transformer. IEEE TIP (2022) 3, 9, 10





33. Liu, Y., Wang, L., Wang, Y., Ma, X., Qiao, Y.: Fineaction: A fine-grained videodataset for temporal action localization. IEEE TIP (2022) 3, 9





34. Loshchilov, I., Hutter, F.: Sgdr: Stochastic gradient descent with warm restarts.arXiv preprint arXiv:1608.03983 (2016) 19





35. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprintarXiv:1711.05101 (2017) 9, 19





36. Ma, N., Zhang, X., Huang, J., Sun, J.: Weightnet: Revisiting the design space ofweight networks. In: ECCV (2020) 3





37. Mac, K.N.C., Joshi, D., Yeh, R.A., Xiong, J., Feris, R.S., Do, M.N.: Learningmotion in feature space: Locally-consistent deformable convolution networks forfine-grained action detection. In: ICCV (2019) 3





38. Meng, Y., Lin, C.C., Panda, R., Sattigeri, P., Karlinsky, L., Oliva, A., Saenko,K., Feris, R.: Ar-net: Adaptive frame resolution for efficient action recognition. In:ECCV (2020) 3





39. Qing, Z., Su, H., Gan, W., Wang, D., Wu, W., Wang, X., Qiao, Y., Yan, J., Gao,C., Sang, N.: Temporal context aggregation network for temporal action proposalrefinement. In: CVPR (2021) 1, 3, 9, 10





40. Ravenscroft, W., Goetze, S., Hain, T.: Deformable temporal convolutional networksfor monaural noisy reverberant speech separation. In: ICASSP (2023) 21





41. Shao, J., Wang, X., Quan, R., Zheng, J., Yang, J., Yang, Y.: Action sensitivitylearning for temporal action localization. In: ICCV (2023) 10, 11





42. Shi, D., Zhong, Y., Cao, Q., Ma, L., Li, J., Tao, D.: Tridet: Temporal actiondetection with relative boundary modeling. In: CVPR (2023) 2, 3, 6, 9, 10, 12, 19,20, 21





43. Shi, D., Zhong, Y., Cao, Q., Zhang, J., Ma, L., Li, J., Tao, D.: React: Temporalaction detection with relational queries. In: ECCV (2022) 3, 10, 19





44. Song, L., Li, Y., Jiang, Z., Li, Z., Sun, H., Sun, J., Zheng, N.: Fine-grained dynamichead for object detection. NeurIPS (2020) 4, 13





45. Sridhar, D., Quader, N., Muralidharan, S., Li, Y., Dai, P., Lu, J.: Class semantics-based attention for action detection. In: ICCV (2021) 1, 3





46. Tan, J., Tang, J., Wang, L., Wu, G.: Relaxed transformer decoders for direct actionproposal generation. In: ICCV (2021) 3, 10





47. Tang, T.N., Kim, K., Sohn, K.: Temporalmaxer: Maximize temporal context withonly max pooling for temporal action localization. arXiv preprint arXiv:2303.09055(2023) 2, 3, 12





48. Wang, L., Huang, B., Zhao, Z., Tong, Z., He, Y., Wang, Y., Wang, Y., Qiao, Y.:Videomae v2: Scaling video masked autoencoders with dual masking. In: CVPR(2023) 10, 11, 19, 20





49. Wang, L., Xiong, Y., Wang, Z., Qiao, Y., Lin, D., Tang, X., Van Gool, L.: Temporalsegment networks: Towards good practices for deep action recognition. In: ECCV(2016) 10, 11





50. Wang, Y., Li, K., Li, Y., He, Y., Huang, B., Zhao, Z., Zhang, H., Xu, J., Liu,Y., Wang, Z., Xing, S., Chen, G., Pan, J., Yu, J., Wang, Y., Wang, L., Qiao, Y.:Internvideo: General video foundation models via generative and discriminativelearning. arXiv preprint arXiv:2212.03191 (2022) 11





51. Wang, Y., Chen, Z., Jiang, H., Song, S., Han, Y., Huang, G.: Adaptive focus forefficient video recognition. In: ICCV (2021) 3





52. Weng, Y., Pan, Z., Han, M., Chang, X., Zhuang, B.: An efficient spatio-temporalpyramid transformer for action detection. In: ECCV (2022) 12





53. Wu, B., Wan, A., Yue, X., Jin, P., Zhao, S., Golmant, N., Gholaminejad, A.,Gonzalez, J., Keutzer, K.: Shift: A zero flop, zero parameter alternative to spatialconvolutions. In: CVPR (2018) 4





54. Wu, Y., He, K.: Group normalization. In: ECCV (2018) 7, 20





55. Wu, Z., Xiong, C., Ma, C.Y., Socher, R., Davis, L.S.: Adaframe: Adaptive frameselection for fast video recognition. In: CVPR (2019) 3





56. Xu, M., Perez Rua, J.M., Zhu, X., Ghanem, B., Martinez, B.: Low-fidelity videoencoder optimization for temporal action localization. In: NeurIPS (2021) 9





57. Xu, M., Zhao, C., Rojas, D.S., Thabet, A., Ghanem, B.: G-tad: Sub-graph local-ization for temporal action detection. In: CVPR (2020) 3, 9, 10, 11





58. Yang, B., Bender, G., Le, Q.V., Ngiam, J.: Condconv: Conditionally parameterizedconvolutions for efficient inference. NeurIPS (2019) 3, 5





59. Yang, L., Han, Y., Chen, X., Song, S., Dai, J., Huang, G.: Resolution adaptivenetworks for efficient inference. In: CVPR (2020) 3, 24





60. Yang, L., Peng, H., Zhang, D., Fu, J., Han, J.: Revisiting anchor mechanisms fortemporal action localization. IEEE TIP (2020) 3, 10





61. Yang, M., Chen, G., Zheng, Y.D., Lu, T., Wang, L.: Basictad: an astoundingrgb-only baseline for temporal action detection. Computer Vision and Image Un-derstanding (2023) 3





62. Zhang, C.L., Wu, J., Li, Y.: Actionformer: Localizing moments of actions withtransformers. In: ECCV (2022) 2, 3, 6, 8, 9, 10, 11, 12, 13, 19, 21, 22





63. Zhao, C., Thabet, A.K., Ghanem, B.: Video self-stitching graph network for tem-poral action localization. In: ICCV (2021) 10, 11





64. Zhao, H., Torralba, A., Torresani, L., Yan, Z.: Hacs: Human action clips and seg-ments dataset for recognition and temporal localization. In: ICCV (2019) 3, 9





65. Zhao, Y., Xiong, Y., Wang, L., Wu, Z., Tang, X., Lin, D.: Temporal action detectionwith structured segment networks. In: ICCV (2017) 9





66. Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., Ren, D.: Distance-iou loss: Faster andbetter learning for bounding box regression. In: AAAI (2020) 8





67. Zhou, C., Loy, C.C., Dai, B.: Interpret vision transformers as convnets with dy-namic convolutions. arXiv preprint arXiv:2309.10713 (2023) 3, 12





68. Zhu, Z., Tang, W., Wang, L., Zheng, N., Hua, G.: Enriching local and globalcontexts for temporal action localization. In: ICCV (2021) 3



# 6 Implementation Details

We now present implementation details including the network architecture, train-ing and inference in our experiments. Further details can be found in our code.

# 6.1 Network architecture

In Fig. 7, we present our network architecture. Specifically, videos will be firstprocessed by a given backbone model to extract the features of the videos. Thenthese pre-extracted features will be used as the inputs of the TAD model. Featurefeature embedding layers based on 2 layers of convolutions followed by LN [3] willfirst used to calculate the input video features. After that, a series of DynE layersare implemented as the encoder, where the first two DynE layers are serviced asthe stem and the rest will successfully downsample the temporal resolution ofthe features will a scale of 2.

Following the common settings in [42, 43, 62], a “2+5” encoder architecturewill be used for each dataset except for Ego4D MQ, meaning that 2 stem layersand 5 downsampling layers will be used for building the encoder, and the outputsof the last 5 layers will be used to build the feature pyramid. Ego4D MQ appliesa “2+7” encoder architecture following the design in [62].

The outputs from the downsampling layers will be first processed by LNs [3]and then used to build the feature pyramid. Then we use separate MSDy-heads for classification and regression as the detectors. In our experiments, the2 adjacent-level features will be sent to the detection head during detection. Forinstance, if we detect the action instances at the 4-th level, then the featuresfrom the 3-rd and 5-th levels will be also sent to the heads. Specially, for the lastlevel, only features from the previous level will be used as inputs. The DyHeadwill share the parameters while detecting the action instances at different featurelevels.

# 6.2 Training details

During training, we randomly selected a subset of consecutive clips from an in-put video and capped the input length to 2304, 768, 960, 2304, and 1024 forTHUMOS14, ActivityNet-1.3, HACS, Epic Kitchens 100, and Ego4D MQV1.0,respectively. Model EMA and gradient clipping are also implemented to fur-ther stabilize the training. We follow the practice in [42, 62] that uses off-the-shelf pre-extracted features as input. Our method is trained with AdamW [35]with warming-up and the learning rate is updated with Cosine Annealing sched-ule [34]. Moreover, hyper-parameters are slightly different across datasets anddiscussed later in our experiment details. More details can be found in our code.We now describe our experiment details for each dataset:

– THUMOS14: We used two-stream I3D [7] pretrained on Kinetics to ex-tract the video features on THUMOS14. VideoMAE V2 [48] is further im-plemented to improve the performance of our method. Following [42], the

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/138109f8b19cd423fe774b3701bdcb5dacf10c0e8a903d498ad9e3ef616c4f88.jpg)



Fig. 7: (a) Overview of DyFADet. (b) The DynE layer consisting of the feature encoder.GN is Group-normalization [54]. (c) The multi-scale feature fusion in DyHead. (d) Theclassification and regression module obtains the classification and boundary results.


initial learning rate is set to $1 e - 4$ with a batch size of 2. We train 40epochs for THUMOS14 containing warmup 20 epochs with a weight decayof $2 . 5 e \mathrm { ~ - ~ } 2$ .

– ActivityNet-1.3 We used TSP features [2] pretrained on Kinetics to extractthe video features. Following [42], the initial learning rate is set to $5 e - 4$We train 15 epochs for ActivityNet-1.3 containing warmup 5 epochs with aweight decay of $5 e - 2$ .

HACS We used SlowFast features [14] pretrained on Kinetics to extract thevideo features on HACS. VideoMAE V2 [48] is also implemented to test theperformance of our method. Following [42], the initial learning rate is setto $5 e - 4$ with a batch size of 8. We train 14 epochs containing warmup 7epochs with a weight decay of $2 . 5 e \mathrm { ~ - ~ } 2$ .

– Epic-Kitchen 100 We used SlowFast features [14] for our method in exper-iments. For both subsets, we train 30 epochs containing warmup 15 epochswith a weight decay of $5 e - 2$ . and the initial learning rate is set to $2 e - 4$with a batch size of 2.

– Ego4D MQv1.0 We used SlowFast features [14] and EgoVLP features [27]in the experiments. For all settings, we train 15 epochs containing warmup5 epochs with a weight decay of $5 e - 2$ . and the initial learning rate is set to$2 e - 4$ with a batch size of 2.

– FineAction We used VideoMAE V2 [48] as the feature extractor for ourmethod. In the experiments, the initial learning rate is set to $5 e - 4$ witha batch size of 8. We train 14 epochs containing warmup 7 epochs with aweight decay of $2 . 5 e \mathrm { ~ - ~ } 2$ .

# 6.3 Inference details

During inference, we fed the full sequence into our model. If a classificationscore is higher than the detection threshold, the instant will be kept. Then, Soft-NMS [4] will be further applied to remove the repeated detected instances. Forour experiments on ActivityNet-1.3 and HACS, we consider score fusion usingexternal classification scores following the settings in [42, 62]. Specifically, givenan input video, the top-2 video-level classes given by external classification scoreswere assigned to all detected action instances in this video, where the actionscores from our model were multiplied by the external classification scores. Eachdetected action instance from our model thus creates two action instances. Moredetails can be found in our code.

# 7 Ablation study of the architecture design

All experiments in this section are conducted on THUMOS14 to explore moreproperties about the DFA and DyFADet.

We provide more comprehensive experimental results to show how the archi-tecture design of the DyFADet can affect its TAD performance. The experimentsinvestigate the effectiveness of different dynamic feature aggregation modules inTAD tasks. The results are shown in Table 10, where different implementationsof the encoder and the detection head are evaluated.

The results in the upper panel of the table are used to evaluate the TADperformance without multi-scale connections in detection heads. The baselinemodel is realized by an all-convolution TAD model, which achieved the mAPof $6 2 . 1 \%$ . We further use the deformable 1-d convolution [40] to substitute theconvolutions in the encoder as the comparison achieves the mAP of 66.1%. Thesuperiority of the deformable-based model demonstrates the strong ability ofdynamic modules in TAD tasks. While, due to the stronger adaptation abilityof DFA, the DyFADet∗ substituting the convolutions with DFA can achieve theperformance equaling to the recent strong TAD model, Actionformer. Moreover,using the proposed DynE layer (DyFADet†) further increases the final perfor-mance by $1 . 0 \%$ .

Moreover, in the middle panel, we see that applying the multi-scale connec-tion in the TAD head can improve the final detection performance. However,naively using the convolution to connect different scales (DyFADet‡) only re-sults in limited improvements. While, after being equipped with DyHead, theDyFADet can achieve a performance of 69.2%, outperforming all other modelsin the experiments. We also use the basic Conv encoder that is built based on allconvolution layers attaching with the proposed DyHead, which achieves the de-tection performance of $6 5 . 8 \%$ , outperforming the baseline model by 3.7%. Sucha result further demonstrates the effectiveness of the proposed DyHead in TADtasks.

In the bottom panel, we test our proposed DyFADet with different dynamicproperties by controlling the $C _ { m }$ as we describe in Section 3 of the main paper.


Table 10: Ablation study for the choices of main components in TAD models. MS-headmeans if using the multi-scale connection in the detection head.


<table><tr><td>Method</td><td>Encoder</td><td>MS-head</td><td>0.3</td><td>0.7</td><td>Avg.</td></tr><tr><td>Baseline</td><td>Conv</td><td>x</td><td>76.3</td><td>40.8</td><td>62.1</td></tr><tr><td>Deformable model</td><td>DeformConv</td><td>x</td><td>81.0</td><td>43.4</td><td>66.1</td></tr><tr><td>DyFADet*</td><td>DFA_Conv</td><td>x</td><td>81.6</td><td>44.6</td><td>66.8</td></tr><tr><td>ActionFormer</td><td>SA</td><td>x</td><td>82.1</td><td>43.9</td><td>66.8</td></tr><tr><td>DyFADet†</td><td>DynE</td><td>x</td><td>83.4</td><td>44.8</td><td>67.8</td></tr><tr><td>Conv Encoder</td><td>Conv</td><td>Dyn</td><td>80.4</td><td>44.2</td><td>65.8</td></tr><tr><td>DyFADet‡</td><td>DynE</td><td>Conv</td><td>83.1</td><td>46.4</td><td>68.0</td></tr><tr><td>DyFADet</td><td>DynE</td><td>Dyn</td><td>84.0</td><td>47.9</td><td>69.2</td></tr><tr><td>DyFADet(k)</td><td>DynE</td><td>Dyn</td><td>82.4</td><td>46.3</td><td>68.4</td></tr><tr><td>DyFADet(ck)</td><td>DynE</td><td>Dyn</td><td>82.7</td><td>47.1</td><td>68.4</td></tr><tr><td>DyFADet(c)</td><td>DynE</td><td>Dyn</td><td>84.0</td><td>47.9</td><td>69.2</td></tr></table>

In our experiments, the $C _ { m }$ is set as the values of $k$ , $C k$ , and $C$ (represented byk, ck, and c in Table 10), where $k$ is the kernel size and the $C$ is the numberof input channels. From the results, we see that the DyFADetc) has the bestperformance on THUMOS14. While other variants of DyFADetan still achievecompetitive detection performance compared to the evaluated TAD models inour main paper.

# 8 Additional Visualizations of DyHead

We provide more visualization results of the proposed DFA to further show howthe proposed dynamic feature learning mechanism can effectively solve the headinadaptation issue in classical TAD models. Following the experiments in themain paper, we use the DyFADet with $C _ { m } = k$ , meaning that the same recep-tive fields will be shared among channels while varying at different timestamps.Noting that although the sampling positions are shared among different chan-nels, the kernel weights will still be adjusted based on the inputs. The resultsare shown in Fig. 8.

From the results we see that the DFA can adapt the re-weight masks basedon inputs, leading to different formations at different timestamps. Noting thatin classical TAD models, such as Actionformer [62], the parameters of the de-tection head are shared among different levels, which might be harmful in de-tection. While, the dynamic properties and multi-scale fusion of the DyFADetspecialize the detection head for inputs and target action instances, leading toa fine-grained dynamic action detection manner that can achieve better TADperformance.

The visualization results also show that the weight of the depth path willalways be used without masking, indicating the importance of the depth paths.This meets our intuition that the depth path is exclusively designed for actiondetection. Moreover, we found that the up paths usually play a more important

![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/624e9a246055316df39a7b471b4073678117490181616c5ca459372043000b46.jpg)



(a) Golf Swing.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/5328052f0014dfc3a14a4ae38956a46253b140445529a15d20a30ec6e4476228.jpg)



(b) Diving.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-01-31/d7b18da8-79e0-423b-aead-17ec4a4e81fb/6be029f2e6416cfc0cb8da4f78728a2a9c28c3ef65ddeb677c2c91476a66e284.jpg)



(c) High Jump.



Fig. 8: Visualization when DyFADet detecting action (a) Golf Swing, (b) Diving and(c) High Jump. The DFA reweight masks at different timestamps at the correspondinglevels in DyHead are shown in the figures. Here, we only represent the masked positionsof the re-weight masks.


role in action detection, which might be because 1) The high-level features areextracted by more encoder layers in the feature encoder, resulting in more high-level semantic information which benefits the detection. 2) Only a few featuresfrom the low-level with high temporal resolution are needed for action detec-tion regardless of the instance duration. 3) The coarse-to-fine feature fusion inDyHead is similar the feature learning process in [59], which demonstrates theimportance of the low-frequency information w.r.t temporal dimension.

We also observe that the later detailed representations generally play a moreimportant role during feature fusion. which might be due to the short-terminformation from the future can be beneficial for predicting the ending of theaction. Moreover, the visualization results show that the predicted action lengthis generally shorter than the ground truth. We infer that this might be dueto that the proposed DyFADet only predicts the intrinsic temporal range ofthe action instance. As shown in in Fig. 8 (a), the DyFADet thinks the actionGolf Swing begins from the time when the person starts to swing the golf club,while ends after the golf club hits the ball. However, the annotation contains thepreparation action and the ending action of the person standing after swinging.Intuitively, both of the temporal ranges can be viewed as the action Golf Swing.The examples in Fig. 6 (b) and Fig. 6 (c) also show the similar trends.