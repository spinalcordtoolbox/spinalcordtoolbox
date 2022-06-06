Introduction to building processing pipelines using scripts
###########################################################

What are batch processing scripts?
----------------------------------

**Batch processing scripts** are scripts written in shell language (i.e. same as your Terminal) containing a sequence of SCT commands. They are meant to establish a processing pipeline, i.e. a chain of commands arranged so that the output of each element is the input of the next.

Why use scripts?
----------------

Scripts are beneficial for a number of reasons:

* They help with **standardization** of processing across subjects.
* They save time through **automation** of processing over large datasets.
* They promote **transparency** for scientific results (e.g., you can publish your article along with the batch processing scripts).
* In the long-term, they help with **reproducibility** by making it possible to re-run entire processing pipelines to regenerate results if needed (e.g., if 6 months after a manuscript submission, the reviewer asks you to change one parameter in the processing).

Example scripts
---------------

The SCT team itself regularly creates customized pipelines for its research efforts. You can see examples of different projects and studies here: https://github.com/sct-pipeline

SCT also provides a large example script called ``batch_processing.sh`` that is meant to introduce all of the main functionality within SCT. You can read more about this on the :ref:`Batch Processing Example <getting-started>` page.

For the purposes of this tutorial, however, we will focus on a smaller script called ``process_data.sh`` that is included alongside the data for this tutorial. The sample script contains a basic processing workflow, demonstrating how to compute CSA for T2 data, and MTR for the white matter regions of MT data.