#ifndef INNER_OUTER_UPDATES_HPP_
#define INNER_OUTER_UPDATES_HPP_

#include "AMReX.H"
#include "AMReX_Geometry.H"

namespace quokka {

inline auto innerUpdateRange(amrex::Box const &validBox, const int nghost)
    -> amrex::Box {
  // check that validBox is big enough
  for (int i = 0; i < AMREX_SPACEDIM; ++i) {
    AMREX_ALWAYS_ASSERT(validBox.length(i) >= 2 * nghost);
  }

  // return interior box for this validBox
  return amrex::grow(validBox, -nghost);
}

inline auto outerUpdateRanges(amrex::Box const &validBox, const int nghost)
    -> std::vector<amrex::Box> {
  // check that validBox is big enough
  for (int i = 0; i < AMREX_SPACEDIM; ++i) {
    AMREX_ALWAYS_ASSERT(validBox.length(i) >= 2 * nghost);
  }

  // return vector of outer boxes for this validBox
  std::vector<amrex::Box> boxes{};

  for (int i = 0; i < 2 * AMREX_SPACEDIM; ++i) {
    // amrex::Box computeRange = amrex::grow(validBox, nghost); // this is wrong
    amrex::Box computeRange = validBox;

    // in 2D, the outer box geometry should look like this
    //  (the x-axis is vertical, and y-axis is horizontal here):
    // 00000000000000000000000000
    // 00000000000000000000000000
    // 222                    333
    // 222                    333
    // 222                    333
    // 11111111111111111111111111
    // 11111111111111111111111111

    switch (i) {
    case 0:
      computeRange.growHi(0, -(computeRange.length(0) - nghost)); // OK
      break;
    case 1:
      computeRange.growLo(0, -(computeRange.length(0) - nghost)); // OK
      break;
    case 2:
      computeRange.grow(0, -nghost);                              // OK
      computeRange.growHi(1, -(computeRange.length(1) - nghost)); // OK
      break;
    case 3:
      computeRange.grow(0, -nghost);                              // OK
      computeRange.growLo(1, -(computeRange.length(1) - nghost)); // OK
      break;
    case 4:
      computeRange.grow(0, -nghost);
      computeRange.grow(1, -nghost);
      computeRange.growHi(2, -(computeRange.length(2) - nghost));
      break;
    case 5:
      computeRange.grow(0, -nghost);
      computeRange.grow(1, -nghost);
      computeRange.growLo(2, -(computeRange.length(2) - nghost));
      break;
    }

    boxes.push_back(computeRange);
  }

  return boxes;
}

} // namespace quokka

#endif // INNER_OUTER_UPDATES_HPP_
