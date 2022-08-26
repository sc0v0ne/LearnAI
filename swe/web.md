# Web Development

<!-- MarkdownTOC -->

- Web Development
- Browser Extensions
- Websites for Developers
- Web Design
- CSS Websites for Developers
  - Animista
  - Clip Path Generator
  - Responsive Font Calculator
  - Type Scale
- CSS Tips
  - Centered
  - Border vs Outline
  - Auto-numbering sections
- References

<!-- /MarkdownTOC -->


## Web Development

- Beautiful Soup
- DevToys
- Django (Python)
- Selenium
- ShareX (Screenshot)

Audio/Video

- LottieFiles
- Remove Bg Video
- Remove Bg Image


## Browser Extensions

- Awesome Screenshots
- Nimbus Screenshot

- Medium Enhanced Stats

- ColorPick
- Colorzilla
- Dark Reader
- Designer Tools
- Eye Dropper
- Fonts Ninja
- Form Vault
- Google Font Previewer
- JsonVue
- Markup Hero
- Momentum
- Octotree
- Site Pallette
- Temp Email
- User JavaScript and CSS
- Wappalyzer
- Web Developer Checklist
- WhatFont
- Window Resizer

UX Design

- Bootstrap Grid Overlay
- ColorTab
- Heurio
- Perfect Pixel
- UX Check

Accessibility

- Spectrum
- Stark
- WAVE


## Websites for Developers

- [Browser frame](https://browserframe.com/)
- [Can I use](https://caniuse.com/?search=Grid)
- [Codepen](https://codepen.io/)
- [DevDocs](https://devdocs.io/)
- [LambdaTest](https://www.lambdatest.com/)
- [Meta Tags](https://metatags.io/)
- [Peppertype](https://www.peppertype.ai/)
- [Profile Pic Maker](https://pfpmaker.com/)
- [Regex101](https://regex101.com/)
- [Resume.io](https:/resume.io)
- [Roadmap](https://roadmap.sh/)
- [Small Dev Tools](https://smalldev.tools/)
- [TypeScript Playground](https://www.typescriptlang.org/)
- [Web Page Test](https://www.webpagetest.org/)



## Web Design

- [Compressor.io](https://compressor.io/)
- [Coolors](https://coolors.co/)
- [CSSIcons](https://css.gg/)
- [Flaticon](https://www.flaticon.com/)
- [Loops](https://loops.wannathis.one/)
- [Ls.Graphics](https://www.ls.graphics)

- [Fontjoy](https://fontjoy.com/6
- [FontShare](https://www.fontshare.com/)
- [Type Scale](https://type-scale.com/)

- [Editor X](https://www.editorx.com/
- [Figma](https://www.figma.com)
- [Mockflow](https://mockflow.com)
- [Mockuuups](https://mockuuups.studio/)
- [Octopus](https://octopus.do/)
- [SipApp](https://sipapp.io/)


## CSS Websites for Developers

- [Animation pausing](https://waitanimate.wstone.uk/)
- [Color Palette Generator](https://mybrandnewlogo.com/color-palette-generator)
- [CSS Generator](https://html-css-js.com/css/generator/box-shadow/)
- [Cubic Bezier Generator](https://cubic-bezier.com/#.17,.67,.83,.67)
- [Gradient Generator](https://cssgradient.io/)
- [Grid Generator](https://cssgrid-generator.netlify.app/)
- [Hamburgers](https://jonsuh.com/hamburgers/)
- [Layout Generator](https://layout.bradwoods.io/)
- [Layout Patterns](https://web.dev/patterns/layout/)
- [Responsively](https://responsively.app/)
- [SchemeColor](https://www.schemecolor.com/)
- [SVG Generator](https://haikei.app/)
- [Transition Animation Generator](https://www.transition.style/)

### [Animista](https://animista.net/)

CSS Animation can get tedious to work with. 

By using Animista, we are able to work interactively with animations.

### [Clip Path Generator](https://bennettfeely.com/clippy/)

```css
  clip-path: polygon(25% 0%, 75% 0%, 100% 53%, 25% 100%, 0% 50%);
```

### [Responsive Font Calculator](https://websemantics.uk/tools/responsive-font-calculator/)

We can easily create a fluid Typography experience which has wider support and can be implemented with a few CSS lines. This experience is just created by using the viewport width, and or height to smoothly scale the root font size. 

We can avoid the jumps that are created by just using media queries.

This web tool will make it for you to fine-tune and design a fluid experience for your users.0

All we have to do is configure the options and you will get a CSS output that you can paste to your side.

```css
  :root {
    font-size: clamp(1rem, calc(1rem + ((1vw - 0.48rem) * 0.6944)), 1.5rem);
    min-height: 0vw;
  }
```

### Type Scale

[Type Scale](https://type-scale.com/)

The fonts are a key aspect of any website, so we have another useful web app to help with fonts. 

When designing a website is it important to see how the different font sizes play together. Using this web app, it is simple to create a consistent font scale.

We can choose between 8 different predetermined scales or build our custom one. We just have to define a growth factor and the tool takes care of the rest.

This will generate fonts using the rem unit it is also handy to see how different base size fonts will look. The default is 16px which matches any browser's default root font.

Once we have everything looking good, we can copy the generated CSS or view the results in a codepen instance. 



## CSS Tips

Here are a few handy CSS tips and features [1]. 

### Centered

Position an element at the center of the screen.  

```css
  .centered {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, 50%);
  }
  
  // position in the center of another element
  .block {
      display: grid;
      place-items: center;
  }
```

### Border vs Outline

The border is inside the element — if we increase the size of the border, then we increase the size of the element. 

The outline is outside the element — if we increase the size of the outline, the element will keep its size and the ribbon around it will grow.

### Auto-numbering sections

We can create a CSS counter and use it in a tag type content, so we can auto-increment a variable and prefix some elements with it. 

This is done using the counter-increment and content properties:



## References

[1] [Some handy CSS tricks](https://medium.com/codex/some-handy-css-tricks-8e5a0d3ac25c)


[10 Google Fonts Every Web Designer Needs To Know](https://uxplanet.org/10-google-fonts-every-web-designer-needs-to-know-de7dc3352d2c)

[3 Best Websites to find Pro and Free Templates for your Portfolio](https://medium.com/geekculture/3-best-websites-to-find-pro-free-templates-for-your-portfolio-c7745792e60)

