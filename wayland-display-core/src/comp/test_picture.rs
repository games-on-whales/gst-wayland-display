//! The `test-picture` property's picture: a fixed 16x9 grid of four solid colours, shown
//! only while no client surface is mapped.
//!
//! A consumer that decodes its own encode of this picture can tell moved pixels from
//! intact ones, which a flat clear colour cannot show. The node agent's media probe holds
//! the other copy of [`GRID`] and of each plane's class order: change them together.

use smithay::utils::{Physical, Point, Rectangle, Size};

pub(crate) const COLS: i32 = 16;
pub(crate) const ROWS: i32 = 9;

/// Class of each cell, rows top to bottom. Neighbours always differ, and the grid matches
/// itself in at most 30% of its cells under any whole-cell shift, so a displaced block
/// lands on other classes.
pub(crate) const GRID: [&[u8; COLS as usize]; ROWS as usize] = [
    b"3010231302302312",
    b"1202103130213231",
    b"2321031023030302",
    b"1230212131212010",
    b"0312101302321203",
    b"1231230121203030",
    b"3023021310132312",
    b"0310303203210120",
    b"2023210310303031",
];

/// sRGB bytes per class. Any two classes are at least 30 code values apart on each of Y, Cb
/// and Cr under BT.601 and BT.709, limited or full range, and rank the same under all four:
/// Y 1<3<0<2, Cb 2<3<0<1, Cr 0<2<1<3. Cb and Cr rank differently, so swapped chroma planes
/// show.
pub(crate) const PALETTE: [[u8; 3]; 4] = [
    [80, 176, 224],
    [128, 32, 224],
    [176, 224, 32],
    [224, 80, 32],
];

/// The cells covering a `width`x`height` frame, with their colours. Edges fall on
/// `i * width / COLS` and `j * height / ROWS`, so the cells tile any size exactly; at
/// 1280x720 they are 80x80, on macroblock and chroma-sample boundaries.
pub(crate) fn cells(
    width: i32,
    height: i32,
) -> impl Iterator<Item = (Rectangle<i32, Physical>, [u8; 3])> {
    (0..ROWS).flat_map(move |row| {
        (0..COLS).map(move |col| {
            let (x0, x1) = (col * width / COLS, (col + 1) * width / COLS);
            let (y0, y1) = (row * height / ROWS, (row + 1) * height / ROWS);
            let class = GRID[row as usize][col as usize] - b'0';
            (
                Rectangle::new(Point::from((x0, y0)), Size::from((x1 - x0, y1 - y0))),
                PALETTE[class as usize],
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_has_no_equal_neighbours_and_no_whole_cell_self_match() {
        let class = |col: i32, row: i32| GRID[row as usize][col as usize];
        for row in 0..ROWS {
            for col in 0..COLS {
                assert!(class(col, row) < b'4');
                assert!(col + 1 == COLS || class(col, row) != class(col + 1, row));
                assert!(row + 1 == ROWS || class(col, row) != class(col, row + 1));
            }
        }
        for shift in 1..COLS - 2 {
            let same = (0..ROWS)
                .flat_map(|row| (shift..COLS).map(move |col| (col, row)))
                .filter(|&(col, row)| class(col, row) == class(col - shift, row))
                .count() as i32;
            assert!(
                same * 10 <= 3 * ROWS * (COLS - shift),
                "column shift {shift}"
            );
        }
        for shift in 1..ROWS - 1 {
            let same = (shift..ROWS)
                .flat_map(|row| (0..COLS).map(move |col| (col, row)))
                .filter(|&(col, row)| class(col, row) == class(col, row - shift))
                .count() as i32;
            assert!(same * 10 <= 3 * COLS * (ROWS - shift), "row shift {shift}");
        }
    }

    #[test]
    fn the_cells_tile_any_frame_size_exactly() {
        for (width, height) in [(1280, 720), (1920, 1080), (1366, 768), (17, 10)] {
            let area: i32 = cells(width, height).map(|(r, _)| r.size.w * r.size.h).sum();
            assert_eq!(area, width * height);
            assert_eq!(cells(width, height).count(), (COLS * ROWS) as usize);
        }
    }
}
