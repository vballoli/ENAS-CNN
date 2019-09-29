# ENAS to find Energy and Speed efficient Neural Network Architectures

Through MacroSearch defined in ENAS, the aim is to include Energy and Speed as reward parameters of the sampled model from the RNN controller to search the Neural Architecture Space of pre defined blocks that include the standard blocks from the paper and additional MobileNet v1 and v2 bottlenecks. 


1. Observe: Model vs Flops vs Inference speed(forward pass)
2. ENAS MacroSearch code - Include standard MacroSearch blocks with MobileNet blocks with intuition derived from the ProxylessNAS paper. 
3. Implement vanilla MacroSearch
4. Include Energy in the MacroSearch
5. Include InferenceTime in the MacroSearch
6. Include both Energy and Time in the MacroSearch (if time permits)
7. Derive Conclusions from the results

