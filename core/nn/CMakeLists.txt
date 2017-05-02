project(nn)

set(nn_headers
  nn.h
  layers.h
  optimizer.h
)

set(nn_sources
  src/nn.cpp
  src/layers.cpp
  src/optimizer.cpp
)

set(nn_libs
  utils
)

add_library(nn STATIC ${nn_sources} ${nn_headers})
target_link_libraries(nn ${nn_libs})

if (BUILD_GO)
  add_executable(nn_go src/nn_go.cpp)
  target_link_libraries(nn_go nn)
  add_executable(som_go src/som_go.cpp)
  target_link_libraries(som_go nn)
endif()

install(TARGETS nn DESTINATION lib)
install(FILES ${nn_headers} DESTINATION include/nn)

install(TARGETS nn
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)