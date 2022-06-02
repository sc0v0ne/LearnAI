# Guide to EthIcal Web Crawling

Web crawling, web scraping, or spider is a computer program technique used to scrape a huge amount of data from websites where regular format data can be extracted and processed into easy-to-read structured formats.

The general belief is that everything you see online is free to scrape and re-use which is probably the biggest misconception regarding web scraping and could land any individual or company in legal hot water.

In short, there are two key web crawler concepts to keep in mind:

1. Ethical Web Crawling

It is considered an unethical practice to set the `user-agent` request header to a common browser UA string (Firefox/Chrome/Edge/Opera) or worse use a rotated list of UA strings and/or IP addresses.
The best practice is to identify your process as a web crawler [1].

2. Law of unintended consequences

An ethical web crawler is actually a very complex software program that takes years to develop and debug to avoid potential side effects on the servers being crawled.

If your web crawler causes problems/issues on web servers it is crawling (which is very common), you could be sued to pay for any costs incurred.

If you follow the unethical approach and do not identify your process as a web crawler, your web crawler could be considered malware and you could face criminal charges. In most cases, the laws and regulations apply to where the server is located, not where you (or your spider) are located.

## Why is Ethical Data Scraping Necessary?

Scraping a single page is pretty straightforward but problems arise when we want to scrape data from a website and collect a large amount of information in a short amount of time.

We can write a crude script that will scrape everything in a fraction of a second, but it will most likely be the last time we get access to the web page in question which is where ethical data scraping comes in handy.

If we respect the fact that a web page has finite resources at its disposal and scrape mindfully, we will most likely not get blocked when web scraping. However, if we want to save ourselves the headache, it is worth looking into web scraping services.

## Potential Pitfalls

There are many potential problems that can arise from your spider. Here we discuss a few of the more common issues:

1. Denial of Service

A major concern by owners of websites is that web crawling may slow down the web server due to repeated requesting of web pages. It may even use up limited bandwidth which is quite similar to a virus attack.

In fact, many internet web servers have less resources than many modern laptops, so it is quite easy for a web crawler to overload a web server's resources.

2. Cost

Webpage crawlers incur costs upon the owner of websites crawled by using up their bandwidth allocation. Keep in mind that different web hosts charge in different ways.
In some cases, you could be required to reimburse website owners for the additional bandwidth costs incurred.

3. Privacy

Everything that appears on the web is in public domain. However, privacy may still be invaded.

4. Copyright

Copyright infringement is one of the most important legal issues for search engines.

Crawlers are involved in illegal activities since they make copies of copyrighted material without the owner's permission (recall napster music controversy).

A particular problem with the internet archive is that it is making web pages freely available for usage. Therefore, owners of websites make use of the robots.txt mechanism to keep their site out of the archive.

## Key Points for Web Crawlers

Things to keep in mind when performing internet scraping [2]:
Take note of terms of use and robots.txt

Before we start web crawling or internet scraping we need to read a website's Terms and Conditions.
It is important to find out if the data is explicitly copyrighted or if any other restrictions are in place.

Most popular and traffic-heavy websites have a robots.txt file that contains a list of instructions for robots or web-crawlers.

The `robots.txt` file can be accessed by appending "/robots.txt" at the end of a website's URL.

2. Identify yourself when sending requests

A user-agent is an HTTP header of a request that is sent to a web page which contains information about the operating system and browser that is being used while visiting a web page.

By default, data scraping scripts should have a unique user-agent identifier that can be easily distinguished from that of a human user since a script does not use a browser.

3. Time-outs and responsive delays

Depending on the amount of bandwidth a website has, we should to be mindful of not overloading their server with our requests.

Multiple, fast-paced requests that are coming from the same IP address and the same user-agent will alert the system administrator that potentially unwanted actions are taking place which will most likely result in a ban.

The simplest way to gather data without overloading a page's server is by setting time-outs.

## Guidelines for Ethical Crawling

Here are some recommendations for ethical crawling [3]:

- Consider the privacy implications of the information collected.

- Consider alternative sources such as Google or Facebook API and internet archives.

- Consider using a commercial or open source web crawling software.

- Be sure to identify your program as a web crawler to web servers.

- Avoid crawling websites for teaching or training purposes.

- Consider the financial costs that may be incurred by website owners from crawling and be prepared to pay for the costs if requested.

- Do not take advantage of naïve site owners who will be unable to identify the causes of bandwidth charges.

- Research and understand the cost implications for crawling web sites.
C
- ontact webmasters of large enterprises to inform them about crawling so that they can opt out.

In summary, there are much more ethical ways to obtain copies of websites (internet archives, APIs, etc.) compared to the risks of using a spider. It is doubtful that "we needed the data" will work as an excuse in a court of law, especially when there are more ethical options available.

## References

[1] [What user agent should I set?](https://webmasters.stackexchange.com/questions/6205/what-user-agent-should-i-set)

[2] [The Ultimate Guide To Ethical Web Scraping](https://finddatalab.com/ethicalscraping)

[3] [Data Crawling Ethics and Best Practices](https://www.promptcloud.com/blog/data-crawling-and-extraction-ethics/)


[Legality of Web Scraping — An Overview](https://medium.com/grepsr-blog/legality-of-web-scraping-an-overview-3cf415885e16)

[Is web crawling legal?](https://towardsdatascience.com/is-web-crawling-legal-a758c8fcacde)

[Appendix C. The Legalities and Ethics of Web Scraping](https://learning.oreilly.com/library/view/web-scraping-with/9781491910283/app03.html#idm139888656381232)

[How to make an ethical crawler in Python](https://dev.to/miguelmj/how-to-make-an-ethical-crawler-in-python-4o1g)

[What is a robots.txt file?](https://moz.com/learn/seo/robotstxt)

[Robots txt File Example: 10 Templates To Use](https://pagedart.com/blog/robots-txt-file-example/)
