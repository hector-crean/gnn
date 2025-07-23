use burn::{
    backend::wgpu::WgpuRuntime,
    tensor::Tensor,
};
use gnn::{
    Graph, GraphNeuralNetwork, GraphNeuralNetworkConfig, NodeFeatures,
};

/// Demonstrate GNN inference for street-level house price prediction
fn gnn_inference<B: burn::tensor::backend::Backend>(device: &B::Device) {
    println!("=== Graph Neural Network Inference Demo ===");
    
    // Create a sample street network
    let graph = Graph::<B>::create_sample_street_network(device);
    
    println!("Created street network with {} nodes (streets)", graph.num_nodes);
    println!("Node features dimension: {}", NodeFeatures::feature_dim());
    
    // Create GNN model
    let config = GraphNeuralNetworkConfig {
        input_dim: NodeFeatures::feature_dim(),
        hidden_dims: vec![32, 16], // Two hidden layers
        output_dim: 5, // 5 house price categories (0=low, 4=high)
    };
    
    let gnn = GraphNeuralNetwork::new(device, config);
    
    // Make predictions
    let predictions = gnn.predict(&graph);
    
    println!("\nGNN Predictions for each street:");
    println!("(Probability distribution over 5 price categories: Low -> High)");
    println!("Predictions shape: {:?}", predictions.dims());
    
    // Get forward pass logits (before softmax)
    let logits = gnn.forward(&graph);
    println!("Raw GNN logits shape: {:?}", logits.dims());
    
    // Show basic information about predictions
    let pred_data = predictions.to_data();
    println!("Sample predictions from first street:");
    let pred_slice = pred_data.as_slice::<f32>().expect("Failed to get prediction data");
    if pred_slice.len() >= 5 {
        println!("First 5 values: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]", 
                 pred_slice[0], pred_slice[1], pred_slice[2], pred_slice[3], pred_slice[4]);
    }
    
    println!("\n=== Graph Structure Analysis ===");
    
    // Show adjacency matrix dimensions
    println!("Adjacency matrix shape: {:?}", graph.adjacency_matrix.dims());
    println!("Node features shape: {:?}", graph.node_features.dims());
    
    // Show normalized adjacency dimensions
    let norm_adj = graph.normalized_adjacency();
    println!("Normalized adjacency shape: {:?}", norm_adj.dims());
    
    // Show some basic statistics
    let adj_sum = graph.adjacency_matrix.sum();
    println!("Total graph edges (sum of adjacency): {:.0}", adj_sum.into_scalar());
    
    // Show sample street network topology
    println!("\nSample Street Network Topology:");
    println!("Street 0 <-> Street 1, Street 2");
    println!("Street 1 <-> Street 0, Street 2, Street 3"); 
    println!("Street 2 <-> Street 0, Street 1, Street 3, Street 4 (central hub)");
    println!("Street 3 <-> Street 1, Street 2, Street 4, Street 5");
    println!("Street 4 <-> Street 2, Street 3, Street 5");
    println!("Street 5 <-> Street 3, Street 4");
    
    if let Some(ref labels) = graph.node_labels {
        println!("\n=== Ground Truth Labels ===");
        println!("Labels shape: {:?}", labels.dims());
        
        let label_data = labels.to_data();
        let label_slice = label_data.as_slice::<f32>().expect("Failed to get label data");
        
        println!("True price categories for each street:");
        let category_names = ["Low", "Low-Med", "Medium", "Med-High", "High"];
        
        // Process each street's label (assuming one-hot encoding)
        for street in 0..graph.num_nodes {
            let start_idx = street * 5; // 5 categories per street
            if start_idx + 4 < label_slice.len() {
                let mut max_idx = 0;
                let mut max_val = label_slice[start_idx];
                
                for i in 1..5 {
                    if label_slice[start_idx + i] > max_val {
                        max_val = label_slice[start_idx + i];
                        max_idx = i;
                    }
                }
                
                println!("Street {}: {} (category {})", street, category_names[max_idx], max_idx);
            }
        }
    }
    
    println!("\n=== GNN Demo Complete ===");
}

/// GNN autodiff demo
fn gnn_autodiff<B: burn::tensor::backend::AutodiffBackend>(device: &B::Device) {
    println!("\n=== GNN Autodiff Demo ===");
    
    // Create a graph with gradient tracking
    let graph = Graph::<B>::create_sample_street_network(device);
    
    // Create GNN model
    let config = GraphNeuralNetworkConfig {
        input_dim: NodeFeatures::feature_dim(),
        hidden_dims: vec![16], // Simpler for demo
        output_dim: 5,
    };
    
    let gnn = GraphNeuralNetwork::new(device, config);
    
    // Forward pass
    let predictions = gnn.forward(&graph);
    
    // Compute a simple loss (mean squared error with dummy targets)
    let dummy_targets = Tensor::zeros_like(&predictions);
    let loss = (predictions - dummy_targets).powf_scalar(2.0).mean();
    
    // Backward pass
    let _gradients = loss.backward(); // Prefix with _ to avoid unused warning
    
    println!("✓ GNN forward and backward passes completed successfully");
    println!("Loss value: {:.6}", loss.into_scalar());
    
    // Note: In a real training setup, we would use these gradients to update model parameters
}

fn main() {
    type MyBackend = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
    type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;
    let device = Default::default();
    
    // Main GNN demonstration
    gnn_inference::<MyBackend>(&device);
    
    // GNN autodiff demonstration  
    gnn_autodiff::<MyAutodiffBackend>(&device);
    
    println!("\n🏠 Graph Neural Network for Street-Level House Price Prediction");
    println!("   Based on the research paper: 'Cities as Graphs'");
    println!("   Implementation uses Burn ML framework with WGPU backend");
    println!("   This demonstrates how street networks can be modeled as graphs");
    println!("   for machine learning applications in urban planning and economics.");
}