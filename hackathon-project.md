# Generating BIDS derivatives with (a) Banana

## Project Description

_Brain imAgiNg Analysis iN Arcana (Banana)_ is a collection of imaging analysis
methods implemented in the [Arcana framework (Close et al. 2019)](), which is
proposed as a code-base for collaborative development of neuroimaging
workflows. Unlike traditional "linear" workflows, analyses implemented in
Arcana are constructed on-the-fly from cascades of modular pipelines that
generate derivatives from a mixture of acquired data and prequisite derivatives
(similar to a Makefile). Given the "data-centric" architecture of this approach,
there should be a natural harmony between it and the ongoing standardisation of
BIDS derivatives.

The primary goal of this project is to closely align the analysis methods
implemented in Banana with the BIDS standard, in particular BIDS derivatives,
in order to make them familiar to new users and interoperable with other
packages. Further to this, in cases where a _de facto_ standard for a particular 
workflow exists (e.g. fmriprep) Banana should aim to mirror this standard by
default. The extensibility of Arcana's object-orientated architecture (via class
inheritance) could then be utilised to tailor such workflows to the needs of
specific studies.

There is also plenty of scope to expand the imaging contrasts/modalities
supported by Banana, so if you have expertise in a particular area and are
interested in implementing it in Banana we can definitely look to do that as
well.  


## Skills required to participate

Any of the following:

* Workflow design + Python (preferably some Nipype but not essential)
* Detailed knowledge BIDS specification (or part thereof)
* Domain-specific knowlege of analysis of a particular imaging modality that
  you would like to see implemented in Banana (e.g. EEG, MEG, etc..)  

## Integration
How would your project integrate a
neuroimager/clinician/psychologist/computational scientist/maker/artist as
collaborator? You can check the Mozilla Open Leadership material on
[personas](https://mozilla.github.io/open-leadership-training-series/articles/building-communities-of-contributors/bring-on-contributors-using-personas-and-pathways/) and [contribution guidelines](https://mozilla.github.io/open-leadership-training-series/articles/building-communities-of-contributors/write-contributor-guidelines/).  
Try to define intermediate goals (milestones).  

## Preparation material

* BioXiv paper
* arcana docs
* banana docs
* nipype docs - Arcana is built on top of Nipype so knowledge of Nipype datatypes is definitely helpful


## Link to your GitHub repo
[Banana GitHub](https://github.com/MonashBI/banana)

[Arcana GitHub](https://github.com/MonashBI/arcana)
  
with [ReadMe.md](https://mozilla.github.io/open-leadership-training-series/articles/opening-your-project/write-a-great-project-readme/) containing  
    &nbsp;&nbsp;&nbsp;&nbsp;* Project idea and context  
    &nbsp;&nbsp;&nbsp;&nbsp;* Installation guidelines if applicable  
    &nbsp;&nbsp;&nbsp;&nbsp;* Links to further reading / tutorials  
and if you want, additional files such as  
    &nbsp;&nbsp;&nbsp;&nbsp;* [Contributors.md](https://mozilla.github.io/open-leadership-training-series/articles/building-communities-of-contributors/write-contributor-guidelines/),
      to specify which types of people can contribute how.

## Communication
Link to the communication channel for your project. You can, for example,
create a [slack channel](https://brainhack-slack-invite.herokuapp.com/) for
your project inside the Brainhack slack community, and include a slack badge
[![slack_brainhack_3](https://user-images.githubusercontent.com/6297454/47951457-5b37b780-df61-11e8-9d77-7b5a4c7af875.png)](https://brainhack-slack-invite.herokuapp.com/)
 to invite people to Brainhack slack, where they can then find and join your
 channel. Or create a community on [gitter](https://gitter.im/) and link to the
 chat by including a Gitter badge linking to your community 
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/yourRoom/Lobby#)