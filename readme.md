
<h1>The Topic Confusion Task - EMNLP 2021 (Findings)</h1>
<h2>Cross-Topic Authorship Attribution</h2>
<table border="1" align="center">
  <tr>
    <td><img src="/images/Picture3.png" alt="" height=160 width=300 /></td>
    <td><img src="/images/Picture1.png" alt="" height=160 width=300 /></td>
    <td><img src="/images/Picture4.png" alt="" height=160 width=300 /></td>
  </tr>
  <tr>
    <td align="center">&nbsp;&nbsp;&nbsp;&nbsp;Same-Topic</td>
    <td align="center">&nbsp;&nbsp;&nbsp;&nbsp;Cross-Topic</td>
    <td align="center">&nbsp;&nbsp;&nbsp;&nbsp;Topic Confusion</td>
  </tr>
</table>

1. Each folder contians Code/*/commandline__.txt which contains commands to run the experiments from cmd.

2. There might be additional packages that should be installed, such as the StanfordNLP tokenizer, and scikit-learn. 

<h2>The Topic Confusion Task</h2>
<table border="1" align="center">
  
  <tr>
    <td><img src="/images/Presentation1.png" alt="" height=400 width=800 /></td>    
  </tr>
</table>

1. main.py has all the baselines/methods from the paper to create Table 2. However, certain sections should be commented out to avoid running the code for too long (magnitude of days).

2. new_model.py is used to create a new model.

<h2>Don't forget to cite our paper!</h2>
@inproceedings{altakrori2021topic,<br>
&nbsp;&nbsp;&nbsp;&nbsp;title={The Topic Confusion Task: A Novel Evaluation Scenario for Authorship Attribution},<br>  
&nbsp;&nbsp;&nbsp;&nbsp;author={Altakrori, Malik and Cheung, Jackie Chi Kit and Fung, Benjamin CM},<br> 
&nbsp;&nbsp;&nbsp;&nbsp;booktitle={Findings of the Association for Computational Linguistics: EMNLP 2021},<br> 
&nbsp;&nbsp;&nbsp;&nbsp;pages={4242--4256},<br> 
&nbsp;&nbsp;&nbsp;&nbsp;year={2021}<br>   
}
