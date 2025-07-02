# Changelog

## v2.0.0

### Added or Changed
- Removed virtual raster windowing that previously clipped input and output to a rectangular region around the Santa Ana River (Mission Blvd to Riverside Dr).  
  ➔ The app now works for areas **outside the Santa Ana River**. However, please note that the segmentation models were trained only in this region, so use in other areas may yield inaccurate results.
- Refactored code for **quicker startup time**.
- Enhanced progress reporting to display **number of tiles processed** and **percentage complete**.

## v1.0.1

### Added or Changed
- Fixed issue where the **close button did not work before running a prediction**.
- Added **GitHub link** to the About window.
- Corrected an error in the **image path in README**.

## v1.0.0

### Added or Changed
- Initial release.
