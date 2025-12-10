# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a project website for "Deep Forcing: Training-Free Long Video Generation with Deep Sink and Participative Compression" - a research paper from KAIST AI. The repository currently contains the project page (HTML/CSS) showcasing research results through videos and interactive comparisons.

## Project Structure

- `index.html` - Main project page with research overview, comparisons, and ablation studies
- `style.css` - Styling for the project page
- `videos/` - Video demonstrations and comparisons
  - `ours_60s/` - 60-second videos generated with Deep Forcing
  - `60s/` - Baseline comparisons (Self Forcing, Rolling Forcing, LongLive)
  - `sinks/` - Ablation study videos showing different sink sizes
- `images/` - Research figures and diagrams
- `README.md` - Basic project information with link to project page

## Website Features

The project page includes:
- Interactive video comparisons with play/pause controls
- Video lightbox for fullscreen viewing
- Prompt display modals showing generation prompts
- Comparison slider for side-by-side baseline comparisons
- Click-to-zoom image lightbox
- MathJax integration for LaTeX rendering

## Development Status

**Note**: The actual Deep Forcing implementation code is not yet available in this repository. The README states "Coming Soon!" for the code release. This repository currently only hosts the project website.

## Website Development

If making changes to the website:
- The page uses vanilla JavaScript (no build step required)
- Videos autoplay on loop and are muted by default
- Custom video controls overlay on hover/tap
- The comparison slider cycles through 4 different prompt comparisons
- Font Awesome icons are loaded from CDN
- Google Fonts (Open Sans, Varela Round) are used for typography

## Links

- Project page: https://cvlab-kaist.github.io/DeepForcing/
- GitHub repository: https://github.com/cvlab-kaist/DeepForcing.git
- Institution: KAIST AI
