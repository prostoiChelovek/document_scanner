cmake_minimum_required(VERSION 3.17)
project(document_scanner)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_search_module(TESSERACT 5 REQUIRED tesseract )
pkg_search_module(LEPTONICA REQUIRED lept )

include_directories(${TESSERACT_INCLUDE_DIRS})
include_directories(${LEPTONICA_INCLUDE_DIRS} )

link_directories(${TESSERACT_LIBRARY_DIRS} )
link_directories(${LEPTONICA_LIBRARY_DIRS} )

add_executable(document_scanner main.cpp)

target_link_libraries(document_scanner ${TESSERACT_LIBRARIES})
target_link_libraries(document_scanner ${LEPTONICA_LIBRARIES} )
target_link_libraries(document_scanner ${OpenCV_LIBRARIES})