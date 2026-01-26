# Student Portfolio Template

This repository provides a simple template for creating a personal data science portfolio website using [Quarto](https://quarto.org/) and [GitHub Pages](https://pages.github.com/). Students will use this template to showcase their projects and learning journey.

## 🚀 Getting Started

### Prerequisites
- [Quarto](https://quarto.org/docs/get-started/) installed
- A GitHub account
- Basic knowledge of Markdown

### Quick Setup

1. **Create your repository**
   - Use this template to create your own repository
   - Name it `your-username.github.io` or any name you prefer
   - Make sure it's **Public**

2. **Customize your site**
   - Replace "Your Name" with your actual name in all files
   - Update your GitHub and LinkedIn links in `_quarto.yml` if you want them.
   - Edit the About Me page with your information

3. **Deploy to GitHub Pages**
   - Run `quarto render` to build your site and `quarto preview` to check it locally
   - Commit and push your changes
   - Enable GitHub Pages: Settings → Pages → Source: Deploy from a branch → Branch: main → Folder: /docs

## 📁 What's Included

```
├── _quarto.yml          # Site configuration
├── index.md             # Your homepage
├── about.md             # About me page
├── projects/            # Your project files
│   ├── eda.qmd          # Data exploration template
│   ├── data-acquisition.qmd  # Data collection template
│   └── final-project.qmd     # Final project template
└── styles.css           # Custom styling
```

## 🛠️ Customization

### Change the Look
- Edit `styles.css` to change colors and fonts
- Modify `_quarto.yml` to change the theme
- Add your own logo or images

### Add Extra Projects
1. Copy files to `projects/`
2. Add it to the navigation in `_quarto.yml`

## 🆘 Need Help?

### Common Issues
- **Site not building**: Make sure all file extensions are correct (.qmd for code, .md for text only)
- **GitHub Pages not updating**: Check that Pages is enabled and pointing to /docs folder
- **Code not running**: Ensure you have Python and required packages installed

### Resources
- [Quarto Documentation](https://https://quarto.org/docs/guide/)
- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Pages Help](https://docs.github.com/en/pages)

---

**Ready to start building your data science portfolio? 🎉**

*This template is designed to be simple enough for beginners while still creating professional-looking results.*

