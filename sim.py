from model import AgentBasedModel

def main():
    model = AgentBasedModel()
    x_over_time, theta_over_time = model.run(seed=0, save_every=10)

    # Print shapes to confirm it worked
    print(x_over_time.shape, theta_over_time.shape)

if __name__ == "__main__":
    main()
