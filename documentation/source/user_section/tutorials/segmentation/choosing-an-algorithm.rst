Choosing between segmentation algorithms
########################################

Although ``sct_deepseg_sc`` was introduced as a follow-up to the original ``sct_propseg``, choosing between the two is not as straightfoward as it may seem. Neither algorithm is strictly superior in all cases; whether one works better than the other is data-dependent. 

As a rule of thumb: 

- ``sct_deepseg_sc`` will generally perform better on "real world" scans of adult humans, including both healthy controls and subjects with conditions such as multiple sclerosis (MS), degenerative cervical myelopathy (DCM), and others. (This is because these kinds of subjects make up the majority of the data used to train ``sct_deepseg_sc``'s underlying model.)
- ``sct_propseg``, on the other hand, will generally perform better on non-standard scans, including exvivo spinal cords, pediatric subjects, and non-human species. (This is because ``sct_propseg`` uses a mesh propagation-based approach that is more agnostic to details such as the shape and size of the spinal cord, the presence of surrounding tissue, etc.)

That said, given the variation in imaging data (imaging centers, sizes, ages, coil strengths, contrasts, scanner vendors, etc.), SCT recommends to try both algorithms with your pilot scans to evaluate the merit of each on your specific dataset, then stick with a single method throughout your study.

Note: Development of these approaches is an iterative process, and the data used to develop these approaches evolves over time. If you have input regarding what has worked (or hasn't worked) for you, we would be happy to hear your thoughts in the `SCT forum <http://forum.spinalcordmri.org/c/sct>`_, as it could help to improve the toolbox for future users.
