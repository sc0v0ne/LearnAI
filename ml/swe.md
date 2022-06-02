# Software Engineering

<!-- MarkdownTOC -->

- Project Quickstart
- SOLID Design Principles
- The Single-responsibility Principle
- SOLID Coding in Python
- Software Engineering Principles
- Changing Software Requirements
- Tips to Design Responsive Website
- References

<!-- /MarkdownTOC -->

## Project Quickstart

1. Who is the user?

  Once you identify the core user, you can identify their core needs.

  If the software does not solve a problem, it doesn’t merit being adopted.

2. How many people will use it?

  Most projects must reach a critical number of users in order to be successful, pay the bills, and get off the ground.

  Choosing the feature that will carry the most impact is essential to making sure the time of the entire team goes into the work with the biggest pay-off.

3. Is it the easiest possible solution?



## SOLID Design Principles

SOLID is an acronym for five software design principles that help make software robust and maintainable. Understanding and applying the five principle is the hallmark for good software engineering. Thus, it is any aspiring software engineer should be acquainted with them.

Cohesion refers to the interconnection between functions in a class. While coupling refers to the interdependence between classes. Source: [1]

The goal is to develop software with high cohesion and low coupling.

High cohesion means we want functions in a class to be performing tasks that are related to each other. 

Low coupling means classes should not depend too much on each other. 

## The Single-responsibility Principle

The single-responsibility principle helps us achieve the goal of high cohesion and low coupling.

A class should only have a single purpose. 

If the functionality of the class needs to be changed there should only be one reason to do so.

## SOLID Coding in Python



## Software Engineering Principles

- Key Principles of Functional Programming for Data Engineering
- Myths Programmers Believe about CPU Caches
- Hints to Avoid Over-complexity in Software Projects
- Understand Temporal Coupling in Code
- Understand Load vs Stress Tests

- Code Smell — When to Refactor
- Design for Services, Not Microservices

- Systems Design Crash Course for ML Engineers
- A Primer on Architectural Patterns
- Speed, Security, and Scalability: Choose 2


- An Introduction to Event-Driven Architecture
- Comparison of the Modes of Event-Driven Architecture



## Changing Software Requirements

When the requirments change, here are a few of the software changes required:

- Code needs to be updated — potentially breaking dependant code
- Integrations updated
- Data updated
- Deployment/DevOps updated
- Data migration updated
- Testing updated
- Documentation updated
- Training updated

A new release through the software lifecycle
The most significant change is extending deadlines and the plan being updated to incorporate all the work needed to make the changes above. When plans change, emotions rise and pressure increases.



## Tips to Design Responsive Website

1. em and rem units instead of px

Always try to use em, percentage, rem Units instead of px for sizing so that the size of text, images etc adjust according to the device

2. Proper use of Margin and Padding

We usually use a lot of padding and margin when we make websites for desktops , to make them more attractive. While making it responsive for mobiles, tablets try decreasing the existing padding and margin

3. Use Box-sizing property

It resolves a lot of problems caused by padding. Using box sizing on HTML elements with a percentage width will take padding into account rather than having to adjust the width because of padding

4. Use flex-box property to align content
Use flexbox to align your HTML elements, such as <div>, <img> etc.It forces elements that can wrap onto multiple lines according to their width

5. Use grid property to design layouts

Use grid property in CSS sheet to create layout of website . Rather than creating extra HTML elements to contain your grid, coloumns and rows, your grid tracks are created within your style sheet

6. Use media query for different screen sizes

Media query should be used to set width and height according to the breakpoints. Breakpoints refer to the width at which the websites look distorted on a particular size of device

7. Use CSS frameworks for Responsive websites

CSS frameworks are great way to build fast and responsive websites.A framework has ready to use code snippets for different purposes. They are very easy to use and embed in your website



## References

[The 3 Questions You Must Ask Before Building a Product](https://medium.com/geekculture/the-3-questions-you-must-ask-before-building-a-product-b2ba104e52f9)


[The Single Responsibility Principle Explained in Python](https://betterprogramming.pub/the-single-responsibility-principle-explained-in-python-622e2d996d86)

[SOLID Coding in Python](https://towardsdatascience.com/solid-coding-in-python-1281392a6a94)

[Code Smell — When to Refactor](https://betterprogramming.pub/code-smell-when-to-refactor-e18f1dca2f01)

[Design for Services; Not Microservices](https://betterprogramming.pub/design-for-services-not-microservices-e339883946d7)


[3 Key Principles of Functional Programming for Data Engineering](https://towardsdatascience.com/3-key-principles-of-functional-programming-for-data-engineering-67d2b82c7483)

[Myths Programmers Believe about CPU Caches](https://software.rajivprab.com/2018/04/29/myths-programmers-believe-about-cpu-caches/)

[10 Hints to Avoid Over-complexity in Software Projects](https://betterprogramming.pub/10-hints-to-avoid-over-complexity-in-software-projects-51a25bf51853)

[Understand Temporal Coupling in Code](https://betterprogramming.pub/temporal-coupling-in-code-e74899f7a48f)

[Understanding Load vs Stress Tests](https://betterprogramming.pub/load-vs-stress-tests-ee49ae110b1d)


[Systems Design Crash Course for ML Engineers](https://towardsdatascience.com/systems-design-crash-course-for-ml-engineers-aafae1cf1890)

[A Primer on Architectural Patterns](https://towardsdatascience.com/a-primer-on-architectural-patterns-fd1b22a4389d)

[Speed, Security, and Scalability: Pick Only 2!](https://betterprogramming.pub/speed-security-and-scalability-pick-only-2-5e61c637b08e)


[An Introduction to Event-Driven Architecture](https://aws.plainenglish.io/event-driven-architecture-2436055f64b1?gi=62c6bfcf207)

[The Comparison of the Modes of Event-Driven Architecture](https://medium.com/geekculture/the-comparison-of-the-modes-of-event-driven-architecture-1742711d79bb)



