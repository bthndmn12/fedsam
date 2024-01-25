import flwr as fl

def main():
        
    # server configuration
    server_config = fl.server.ServerConfig(
        num_rounds=10 
    )

    # strategy selection
    strategy = fl.server.strategy.FedAvg(
    
        min_fit_clients=2, 
        min_available_clients=2  
    )

    # server initialization
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
            strategy=strategy
    )

if __name__ == "__main__":
    main()
