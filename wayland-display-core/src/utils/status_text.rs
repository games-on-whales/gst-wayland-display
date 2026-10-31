//! The status panel the compositor shows while no application window is on screen.
//!
//! A session's stream starts before its application has a window: the app's container may
//! still be downloading or starting. Without anything to draw, the client would see a black
//! screen with no idea what is happening. The host sets a status line ("Downloading
//! Steam... 43%") with [`crate::WaylandDisplay::set_status`]; the compositor draws it
//! centred on the output until an application window is mapped.
//!
//! Text is rasterised with the public-domain `font8x8` glyphs (no font files or system
//! libraries needed), scaled with the output height so it stays legible on a TV.

use font8x8::{BASIC_FONTS, LATIN_FONTS, UnicodeFonts};

/// Longest line before wrapping, in characters (fewer when the output is narrower).
const MAX_COLUMNS: usize = 48;
/// Glyph cells of padding around the text.
const PADDING_CELLS: usize = 2;
/// Premultiplied ABGR8888 (byte order R, G, B, A) of the text and the panel behind it.
const TEXT_RGBA: [u8; 4] = [255, 255, 255, 255];
const PANEL_RGBA: [u8; 4] = [0, 0, 0, 200];

/// A rasterised status panel: premultiplied `Abgr8888` pixels of `size` (width, height).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusImage {
    pub pixels: Vec<u8>,
    pub size: (i32, i32),
}

/// Pixels per font pixel for an output `height` pixels tall: 3 at 1080p, at least 2.
pub fn scale_for(height: u32) -> usize {
    ((height / 360) as usize).max(2)
}

/// `text` split into lines of at most `max_columns` characters, breaking at spaces where
/// possible. Explicit newlines are kept.
pub fn wrap(text: &str, max_columns: usize) -> Vec<String> {
    let max_columns = max_columns.max(1);
    let mut lines = Vec::new();
    for paragraph in text.lines() {
        let mut line = String::new();
        for word in paragraph.split(' ') {
            let mut word: Vec<char> = word.chars().collect();
            // A word longer than a line is broken across lines.
            while word.len() > max_columns {
                if !line.is_empty() {
                    lines.push(std::mem::take(&mut line));
                }
                lines.push(word.drain(..max_columns).collect());
            }
            let word: String = word.into_iter().collect();
            let len = line.chars().count();
            if len > 0 && len + 1 + word.chars().count() > max_columns {
                lines.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(&word);
        }
        lines.push(line);
    }
    lines
}

fn glyph(c: char) -> [u8; 8] {
    BASIC_FONTS
        .get(c)
        .or_else(|| LATIN_FONTS.get(c))
        .or_else(|| BASIC_FONTS.get('?'))
        .unwrap_or([0; 8])
}

/// Rasterise `text` for a `width` x `height` output, no wider than the output. `None` for
/// empty text.
pub fn render(text: &str, width: u32, height: u32) -> Option<StatusImage> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let scale = scale_for(height);
    let cell = 8 * scale;
    let fit = (width as usize / cell).saturating_sub(2 * PADDING_CELLS);
    let lines = wrap(text, MAX_COLUMNS.min(fit));
    // One and a half cells per line, so lines don't touch.
    let line_height = cell + cell / 2;
    let columns = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0);
    let width = (columns + 2 * PADDING_CELLS) * cell;
    let height_px = lines.len() * line_height - cell / 2 + 2 * PADDING_CELLS * cell;

    let mut pixels = PANEL_RGBA.repeat(width * height_px);
    for (row, line) in lines.iter().enumerate() {
        // Centre each line in the panel.
        let line_width = line.chars().count() * cell;
        let x0 = (width - line_width) / 2;
        let y0 = PADDING_CELLS * cell + row * line_height;
        for (i, c) in line.chars().enumerate() {
            for (gy, bits) in glyph(c).iter().enumerate() {
                for gx in 0..8 {
                    if bits & (1 << gx) == 0 {
                        continue;
                    }
                    for sy in 0..scale {
                        let y = y0 + gy * scale + sy;
                        let start = (y * width + x0 + i * cell + gx * scale) * 4;
                        for px in pixels[start..start + scale * 4].as_chunks_mut::<4>().0 {
                            *px = TEXT_RGBA;
                        }
                    }
                }
            }
        }
    }
    Some(StatusImage {
        pixels,
        size: (width as i32, height_px as i32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_text_wraps_at_word_boundaries() {
        let lines = wrap(
            "Downloading ghcr.io/games-on-whales/steam:edge 1.2 GB of 3.4 GB",
            MAX_COLUMNS,
        );
        assert!(lines.iter().all(|l| l.chars().count() <= MAX_COLUMNS));
        assert_eq!(
            lines.join(" "),
            "Downloading ghcr.io/games-on-whales/steam:edge 1.2 GB of 3.4 GB"
        );
        assert_eq!(wrap("one\ntwo", MAX_COLUMNS), vec!["one", "two"]);
    }

    #[test]
    fn a_word_longer_than_a_line_is_broken() {
        let word = "x".repeat(MAX_COLUMNS * 2 + 3);
        let lines = wrap(&word, MAX_COLUMNS);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines.concat(), word);
    }

    #[test]
    fn panel_has_text_pixels_on_a_dark_background() {
        let image = render("Starting Steam...", 1920, 1080).unwrap();
        let (w, h) = image.size;
        assert_eq!(image.pixels.len(), (w * h * 4) as usize);
        let text = image
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| **p == TEXT_RGBA)
            .count();
        let panel = image
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| **p == PANEL_RGBA)
            .count();
        assert!(text > 0 && panel > text, "text {text} panel {panel}");
        // Scaled for a TV: 3 output pixels per font pixel at 1080p.
        assert_eq!(h as usize, 8 * 3 + 2 * PADDING_CELLS * 8 * 3);
    }

    #[test]
    fn empty_text_draws_nothing() {
        assert_eq!(render("  ", 1920, 1080), None);
    }

    #[test]
    fn panel_fits_a_narrow_output() {
        let text = "Downloading ghcr.io/games-on-whales/steam:edge, 1.2 GB of 3.4 GB";
        let image = render(text, 640, 480).unwrap();
        assert!(image.size.0 <= 640, "{} px wide", image.size.0);
    }
}
