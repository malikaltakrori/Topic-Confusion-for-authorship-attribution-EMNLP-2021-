
<h1>The Topic Confusion Task - EMNLP 2021 (Findings)</h1>
<h2>Cross-Topic Authorship Attribution</h2>
<table border="1" align="center">
  <tr>
    <td><img src="/images/Picture3.png" alt="" height=160 width=300 /></td>
    <td><img src="/images/Picture1.png" alt="" height=160 width=300 /></td>
    <td><img src="/images/Picture4.png" alt="" height=160 width=300 /></td>
  </tr>
  <tr>
    <td align="center">Same-Topic</td>
    <td align="center">Cross-Topic</td>
    <td align="center">Topic Confusion</td>
  </tr>
</table>




1. Each folder contians Code/*/commandline__.txt which contains commands to run the experiments from cmd.

2. There might be additional packages that should be installed, such as the StanfordNLP tokenizer, and scikit-learn. 

<h2>The Topic Confusion Task</h2>
<table border="1" align="center">
  
  <tr>
    <td><img src="/images/Presentation1.png" alt="" height=300 width=450 /></td>    
  </tr>
</table>

1. main.py has all the baselines/methods from the paper to create Table 2. However, certain sections should be commented out to avoid running the code for too long (magnitude of days).

2. new_model.py is used to create a new model.

