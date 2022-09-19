# MongoDB

[MongoDB Tutorial](https://www.geeksforgeeks.org/mongodb-tutorial/)

```bash
  mongosh "mongodb://mongodb.cogentcoder.com:27107" --username mongoroot --authenticationDatabase admin
```

```sql
  mongosh "mongodb://localhost/admin"
  mongosh "mongodb://localhost:27017/admin" --username mongoroot
  mongosh "mongodb://localhost:27017/admin" --username mongoroot --authenticationDatabase admin

  mongosh "mongodb+srv://mongodb.cogentcoder.com/"
  mongosh "mongodb://mongodb.cogentcoder.com:27017/admin" --tls --username mongoroot --authenticationDatabase admin
  
  show dbs

  # list all collections in the records database
  use flaskdb
  db.getCollectionInfos()
  db.inventory.getIndexes()
```

## Fix TTL Index Issue

```sql
  db.eventlog.createIndex( { "lastModifiedDate": 1 }, { expireAfterSeconds: 3600 } )

  db.log_events.createIndex( { "createdAt": 1 }, { expireAfterSeconds: 10 } )
  db.log_events.insertOne({
   "createdAt": new Date(),
   "logEvent": 2,
   "logMessage": "Success!"
  })

```