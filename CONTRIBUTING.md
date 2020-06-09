# Contributing to AI-JACK

In this document you can find rough guidelines of how to contribute to AI-JACK. This is a guideline, not a set of rules. Feel free to propose changes to this document in a pull request.

## Rewards

Contributions are rewarded! If you provide a valuable enhancement, bug fixes, or suggest new useful features to AI-JACK, we'll send you AI-JACK stickers with a small gift :) 

Thank you for your interest! In case of any questions, submit a new issue or contact ai.jack@bilot.fi.

## How to contribute?

### Reporting Bugs

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). If you notice any bug, create an issue on AI-jack repository and provide all the necessary information.

Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a proper label** to help others identify type of an issue.
* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible. 
* **Provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples. If you're providing snippets in the issue, use [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Explain which behavior you expected to see instead and why.**
* **Include screenshots** which show you following the described steps and clearly demonstrate the problem. 
* **If the problem wasn't triggered by a specific action**, describe what you were doing before the problem happened and share more information using the guidelines below.

Provide more context by answering these questions:

* **Did the problem start happening recently** (e.g. after a release update) or was this always a problem?
* **Can you reliably reproduce the issue?** If not, provide details about how often the problem happens and under which conditions it normally happens.
* **Which version of R are you using?**
* **What's the name and version of the OS you're using**?
* **Which packages do you have installed?**

### Suggesting Enhancements

If you would like AI-jack to be better, feel free to propose enhancements! Those may include adding new machine learning models, evaluation metrics, integration or just code enhancements for better performance.

Before creating enhancement suggestions, please make sure that this enhancement is not available already. If you are unsure, you can check the manual.

Enhancement suggestions are tracked as [GitHub issues](https://guides.github.com/features/issues/), similar as bugs. When submitting a new issue, please kindly provide following details:

* **Use 'enhancement' label**
* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include copy/pasteable snippets which you use in those examples, as [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.

### Code Contribution

You are very welcome to contribute to AI-jack. If you are unsure where to start, first have a look at [open issues](https://github.com/Bilot/AI-jack-opensource-R/issues). A bunch of labels is available to browse the issues.

### Pull Requests

As we would like to review your pull request as quickly and smoothly as possible, please follow these steps to have your contribution considered by the maintainers:

* Follow the styleguide described below when coding.
* Test your code on the sample data from repository.
* If you are fixing a bug, link to the issue describing this bug.
* If you are improving performance, what are quantitative benefits (reduced time, memory use etc.) from this enhancement? 
* Describe what have you changed. Note that this may require updating README as well to update our manual. Use labels!
* Mention any possible drawbacks/inconveniences that your change may imply.
* Ensure any install or build dependencies are removed before the end of the layer when doing a build.
* Most importantly, verify that all [status checks](https://help.github.com/articles/about-status-checks/) are passing. In case they are not and you are not able to find any mistake in your change, please leave a comment with details.

While the prerequisites above must be satisfied prior to having your pull request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes before your pull request can be ultimately accepted.

## R Styleguide

This guideline set is based mostly on [general R styleguide](http://adv-r.had.co.nz/Style.html) and is open for any suggestions via issues.

* Set parameter defaults with spaces around the equal sign
    * `train_glm_model(model_name = ii,
                               set = set)`
       instead of `train_glm_model(model_name=ii,
                               set=set)`
* Place spaces around all infix operators (=, +, -, <-, etc., **except** `:` or `::`).
    * `i + 1` instead of `count+1`, but `x <- 1:10` and `base::get`
* No need to place a space before left parentheses, either in function call or otherwise.
* Do not place spaces around code in parentheses or square brackets (unless there’s a comma).
    * `x[2, ]` instead of `x[2,]` or `x[2 , ]`
* Keep in mind that an opening curly brace should be followed by a new line. A closing curly brace also needs a separate line.
* Limit your line length to 80 characters at max. If your lines are longer, please consider encapsulating it in a function. 
* Add as much comments as you can. A comment section should start with a title line. Important titles should be capitalised.
* Names of modules, functions or function parameters should be lower-cased, with one or two words/abbreviations. If two, separate them with '_' underline. If adding code to existing module, follow general patterns of the module.

And **don't hesitate to contact us**! We are excited to work with you :)
