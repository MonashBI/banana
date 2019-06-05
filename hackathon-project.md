# Generating BIDS derivatives with (a) Banana

## Project Description

_Brain imAgiNg Analysis iN Arcana (Banana)_ is a collection of imaging analysis
methods implemented in the [Arcana framework](https://www.biorxiv.org/content/10.1101/447649v3), which is
proposed as a code-base for collaborative development of neuroimaging
workflows. Unlike traditional "linear" workflows, analyses implemented in
Arcana are constructed on-the-fly from cascades of modular pipelines that
generate derivatives from a mixture of acquired data and prequisite derivatives
(similar to Makefiles). Given the "data-centric" architecture of this approach, there should be a natural harmony between it and the ongoing standardisation of BIDS derivatives.

The primary goal of this project is to closely align the analysis methods
implemented in Banana with the BIDS standard, in particular BIDS derivatives,
in order to make them familiar to new users and interoperable with other
packages. Further to this, in cases where a _de facto_ standard for a particular 
workflow exists (e.g. fmriprep) Banana should aim to mirror this standard by
default. The extensibility of Arcana's object-orientated architecture (via class
inheritance) could then be utilised to tailor such standard workflows to the needs of
specific studies.

There is also plenty of scope to expand the imaging contrasts/modalities
supported by Banana, so if you have expertise in a particular area and are
interested in implementing it in Banana we can definitely look to do that as
well. 

## Skills required to participate

***Any*** of the following:

* Python
* Workflow design (preferably some Nipype but not essential)
* Detailed knowledge BIDS specification (or part thereof)
* Domain-specific knowlege of analysis of a particular imaging modality that
  you would like to see implemented in Banana (e.g. EEG, MEG, etc..)

## Integration

* Neuroinformaticians who are looking to implement and maintain a suite of generic analysis methods
* PhD students who are looking to design a comprehensive analysis for their thesis
* Domain-experts who a looking to implement their existing workflows in a portable framework

Try to define intermediate goals (milestones). 

## Preparation material

Skimming through the Arcana paper to get up to speed on the concepts would be a good idea.

[Arcana BioXiv paper](https://www.biorxiv.org/content/10.1101/447649v3) (_in press_  _Neuroinformatics_, to be [10.1007/s12021-019-09430-1](https://doi.org/10.1007/s12021-019-09430-1))

There is also some online documentation, but the paper is more comprehensive at this stage

[arcana docs](http://arcana.readthedocs.io)

Arcana is built on top of Nipype, so if you want to get your hands dirty implementing some analyses understanding its concepts is also important.

[nipype docs](https://nipype.readthedocs.io)

## Link to your GitHub repo

[Banana Github Repo](https://github.com/MonashBI/banana)

## Communication

I have set up a new channel on the BrainHack mattermost [here](https://mattermost.brainhack.org/brainhack/channels/banana)