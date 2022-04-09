
<h1>The Topic Confusion Task - EMNLP 2021 (Findings)</h1>

1. [Cross Topic/commandline__.txt](/Cross%20Topic/commandline%20to%20call%20the%20code.txt) contains commands to run the experiments from cmd.

2. There might be additional packages that should be installed, such as the StanfordNLP tokenizer, and scikit-learn. 

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

<h2>The Topic Confusion Task</h2>
<table border="1" align="center">
  
  <tr>
    <td><img src="/images/Presentation1.png" alt="" height=400 width=800 /></td>    
  </tr>
</table>

1. [Topic confusion/commandline.txt](/Topic%20Confusion/commandline.txt) contains commands to run the experiments from cmd.
 
2. [Topic confusion/main.py](/Topic%20Confusion/main.py) has all the baselines/methods from the paper to create Table 2. However, certain sections should be commented out to avoid running the code for too long (magnitude of days).

3. [Topic confusion/new_model.py](/Topic%20Confusion/new_model.py) Topic is used to create a new model.

<h2>Data</h2>
We do not have the rights to share the actual data as per the Guardian API policy. Detailed instructions to collect the data can be found <a href=https://malikaltakrori.github.io/publications/TCT/>here</a>

<h2>Don't forget to cite our paper!</h2>
@inproceedings{<b>altakrori2021topic</b>,<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>title</b>={The Topic Confusion Task: A Novel Evaluation Scenario for Authorship Attribution},<br>  
&nbsp;&nbsp;&nbsp;&nbsp;<b>author</b>={Altakrori, Malik and Cheung, Jackie Chi Kit and Fung, Benjamin CM},<br> 
&nbsp;&nbsp;&nbsp;&nbsp;<b>booktitle</b>={Findings of the Association for Computational Linguistics: EMNLP 2021},<br> 
&nbsp;&nbsp;&nbsp;&nbsp;<b>pages</b>={4242--4256},<br> 
&nbsp;&nbsp;&nbsp;&nbsp;<b>year</b>={2021}<br>   
}
