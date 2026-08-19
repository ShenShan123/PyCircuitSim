cmake_minimum_required(VERSION 3.20)
project(bsim_cmg_osdi)

set(BSIMCMG_VA "${CMAKE_SOURCE_DIR}/bsim-cmg-va/code/bsimcmg_main.va")
set(BSIMCMG_INC_DIR "${CMAKE_SOURCE_DIR}/bsim-cmg-va/code")
set(OSDI_OUT_DIR "${CMAKE_BINARY_DIR}/osdi")
set(OSDI_OUT "${OSDI_OUT_DIR}/bsimcmg.osdi")

find_program(OPENVAF_BIN NAMES openvaf HINTS /usr/local/bin)
if(NOT OPENVAF_BIN)
  message(FATAL_ERROR "openvaf not found. Set OPENVAF_BIN or install OpenVAF.")
endif()

file(MAKE_DIRECTORY "${OSDI_OUT_DIR}")

add_custom_command(
  OUTPUT "${OSDI_OUT}"
  COMMAND "${OPENVAF_BIN}" -I "${BSIMCMG_INC_DIR}" -o "${OSDI_OUT}" "${BSIMCMG_VA}"
  DEPENDS "${BSIMCMG_VA}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMENT "Compiling BSIM-CMG Verilog-A to OSDI: ${OSDI_OUT}"
  VERBATIM
)

add_custom_target(osdi ALL DEPENDS "${OSDI_OUT}")

install(FILES "${OSDI_OUT}" DESTINATION osdi)
