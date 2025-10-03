for i in {1..32}
do
cat <<EOF > client-job-$i.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: client-job-$i
spec:
  template:
    metadata:
      labels:
        app: client-$i
    spec:
      containers:
      - name: client
        image: ccsaba99/client-image:latest  # Replace with your Docker image
        env:
        - name: CLIENT_NUM
          value: "$i"  # Dynamically set CLIENT_NUM based on the iteration
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb-secret   # <-- matches the secret name
              key: WANDB_API_KEY   # <-- the key inside that secret        
        volumeMounts:
        - mountPath: "/app/logs"  # Path inside the container where the volume will be mounted
          name: storage-volume          
      restartPolicy: Never  # Ensures the Job does not restart after completion
      volumes:
      - name: storage-volume
        persistentVolumeClaim:
          claimName: my-nfs-pvc  # Name of the PVC defined above      
  backoffLimit: 0  # Prevent retries on failure
EOF
done
