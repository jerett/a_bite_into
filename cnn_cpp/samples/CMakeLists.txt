add_executable(DemoCheckGradient "DemoCheckGradient.cc")
target_link_libraries(DemoCheckGradient SrcLib ${OpenCV_LIBS})

add_executable(DemoMnist "DemoMnist.cc")
target_link_libraries(DemoMnist SrcLib ${OpenCV_LIBS})

add_executable(DemoCifar10 "DemoCifar10.cc")
target_link_libraries(DemoCifar10 SrcLib ${OpenCV_LIBS})

add_executable(DemoBinaryClassification "DemoBinaryClassification.cc")
target_link_libraries(DemoBinaryClassification SrcLib ${OpenCV_LIBS})