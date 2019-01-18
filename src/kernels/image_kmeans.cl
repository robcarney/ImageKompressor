__kernel void FindClosestCentroid(
    __global float* input,
    __constant float* centroids,
    __global int* indices)
{
    const int2 pos = { get_global_id(0), get_global_id(1) };
    int imgIdx = (pos.x * 3) + (IMG_WIDTH * 3 * pos.y);

    float r = input[imgIdx];
    float g = input[imgIdx + 1];
    float b = input[imgIdx + 2];
    float4 currPixelValue = (float4) (r, g, b, 0.f);

    float min = 1.0f;
    int closestCentroidIndex = -1;
    int count = 0;
    for (int i = 0; i < NUM_CENTROIDS; i++)  {
        float4 currCent = (float4) (centroids[i*3], centroids[i*3+1], centroids[i*3+2], 0.f);
        float dist = distance(currPixelValue, currCent);
        if (dist < min)  {
            min = dist;
            closestCentroidIndex = i;
        }
        count++;
    }
    int idx = imgIdx / 3;
    indices[idx] = closestCentroidIndex;
}