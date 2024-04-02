run_command silo_observer "run silo_observer static_base_link_broadcaster base_link 0 0 0 0 0 0"
run_command silo_observer "run silo_observer silo_coordinates_broadcaster"
run_command silo_observer "run silo_observer silo_observer_node"